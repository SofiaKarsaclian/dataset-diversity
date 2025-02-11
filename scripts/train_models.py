import logging
import warnings
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pickle
import transformers
from datasets import Dataset
from huggingface_hub import HfFolder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from utils.model_utils import (
    compute_metrics,
    compute_metrics_hf
)

transformers.logging.set_verbosity(transformers.logging.ERROR)
logging.disable(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

base_model = "roberta-base"
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # dynammically pads sequences to the length of the longest sequence in each batch

def ensure_long_tensor(examples):
        examples["label"] = torch.tensor(examples["label"]).long()  # Convert labels to LongTensor
        return examples

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=128, return_tensors="pt")

def convert_to_hf_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_function, batched=True) 
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    dataset = dataset.map(ensure_long_tensor, batched=True)

    return dataset


def train_subsamples(subsamples_dataset, dataset_name, test_set, dev_set=None):
    results_list = []
    output_dir = f"./results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the Hugging Face token from the environment
    hf_token = HfFolder.get_token()

    for name, subset in subsamples_dataset.items():
      for subsample_name, data in subset.items():
        df = data["data"]
        df_vs = data.get("vs", None)

        if dev_set is None:
            # If no dev set is explicitly provided, split test_set to create one
            dev_df, test_df = train_test_split(test_set, test_size=0.67, random_state=42, stratify=test_set["label"])
        else:
            dev_df = dev_set
            test_df = test_set

        train_df = convert_to_hf_dataset(df, tokenizer)
        dev_df = convert_to_hf_dataset(dev_df, tokenizer)
        test_df = convert_to_hf_dataset(test_df, tokenizer)

        # Define a unique Hugging Face repo name
        hf_model_id = f"{name}_model_{subsample_name}"

        training_args = TrainingArguments(
            output_dir=output_dir, 
            per_device_eval_batch_size=32,
            per_device_train_batch_size=32,
            num_train_epochs=3,
            save_total_limit=1,
            evaluation_strategy="steps",
            logging_steps=50,
            eval_steps=50,
            save_steps=50,
            disable_tqdm=False,
            weight_decay=0.05,
            learning_rate=2e-5,
            run_name=hf_model_id,
            metric_for_best_model="eval_loss",
            save_strategy="steps",
            load_best_model_at_end=True,
            remove_unused_columns=False,
            push_to_hub=True,  # Push to Hugging Face Hub
            hub_model_id=hf_model_id,  # Unique model identifier on Hugging Face Hub
            hub_token=hf_token,  # Hugging Face token for authentication
        )

        # Trainer initialization
        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_df,
            eval_dataset=dev_df,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_hf,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=5,
                    early_stopping_threshold=0.0,
                ),
            ],
        )

        # Train the model
        trainer.train()

        # Push the trained model to the Hugging Face Hub
        trainer.push_to_hub(commit_message=f"Model trained on {name}")

        # Evaluate on the test set
        eval_dataloader = DataLoader(
            test_df,
            batch_size=32,
            collate_fn=data_collator,
            pin_memory=True,
        )

        # Compute metrics
        metrics = compute_metrics(eval_dataloader, model)
        metrics['subsample'] = name
        if df_vs is not None:
            metrics['vs'] = df_vs
        results_list.append(metrics)

        # Save metrics locally
        results_df = pd.DataFrame(results_list)
        results_df = results_df[['subsample'] + [col for col in results_df.columns if col != 'subsample']]
        results_df.to_csv(f"results_{dataset_name}_df.csv", index=False)

    return results_df


if __name__ == "__main__":
    
    subsample_dir = 'data/subsamples'
    subsamples_dataset = {}
    
    for filename in os.listdir(subsample_dir):
        if filename.endswith(".pkl"):
            file_path = os.path.join(subsample_dir, filename)
            with open(file_path, "rb") as f:
                subsample_name = filename.replace(".pkl", "")
                subsamples_dataset[subsample_name] = pickle.load(f)

            dataset_name = subsample_name

            # Select the test set based on the pickle filename 
            if "babe" in filename.lower() or "annomatic" in filename.lower():
                test_set = pd.read_parquet('data/processed/test_sets/babe_test.parquet')
            elif "basil" in filename.lower():
                test_set = pd.read_parquet('data/processed/test_sets/basil_test.parquet')
            else:
                raise ValueError(f"Unknown test set for file: {filename}")

            # Train on subsamples
            train_subsamples(subsamples_dataset, dataset_name, test_set)