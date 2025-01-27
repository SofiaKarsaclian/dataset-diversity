from abc import ABC, abstractmethod
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import os
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd
from checklist.utils import get_model_name

class BaseTest(ABC):
    """
    Base class for test cases.

    This class defines the common interface for test cases and provides
    default implementations for some of the methods.

    Attributes:
    - data_path: The path to the test data.
    - test_data: The test data for the current test case.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.test_data = None

    def make_predictions(self, data=None, target_col: str = "text"):
        """
        Generates predictions for the test data using the trained model.

        Returns:
            predictions (list): List of predicted labels for the test data.
        """
        if data is None:
            data = self.test_data
        tok = self.tokenizer(
            list(data[target_col]),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        testing_dataloader = DataLoader(
            Dataset.from_dict(tok), batch_size=32, collate_fn=data_collator
        )

        predictions = []
        for batch in tqdm(testing_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())

        return predictions

    def compute_metrics(self, y_true, y_preds):
        """
        Compute the Matthews correlation coefficient (MCC) between the true labels and predicted labels.

        Parameters:
        - y_true (array-like): The true labels.
        - y_preds (array-like): The predicted labels.

        Returns:
        - mcc (float): The Matthews correlation coefficient.
        """
        return matthews_corrcoef(y_true, y_preds)

    def initialize_model(self, model_checkpoint: str):
        """
        Initializes the model for sequence classification.

        Args:
            model (str): The name or path of the pre-trained model.

        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)

    @abstractmethod
    def prepare_test_data(self):
        """
        Prepares the test data for the current test case.

        This method should be implemented by subclasses to provide the necessary test data
        for the specific test case.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, test_data) -> pd.DataFrame:
        """
        This method is used to test the given model.

        Parameters:
        - model: The model to be tested.

        Returns:
        - pd.DataFrame with predictions
        """
        raise NotImplementedError

    
    def execute(self, model_checkpoint: str, results_store: list):
        """
        Executes the test on the given model and appends results to a shared list.

        Args:
            model_checkpoint (str): The model to be tested.
            results_store (list): A list to store results for all models.

        Returns:
            None
        """
        if self.test_data is None:
            print("Preparing test data...")
            self.prepare_test_data()

        self.initialize_model(model_checkpoint=model_checkpoint)

        print("Running model on the test...")
        result = self.test(test_data=self.test_data.copy())  # The test method adds predictions

        test_type = self.__class__.__name__  # Get the test type dynamically
        file_path = "checklist/data/results.csv"

        # Load existing results if the file exists
        existing_results = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()

        if test_type == "PrejudiceTest":
            for category in result["category"].unique():
                category_data = result[result["category"] == category]

                for category_minority in category_data["category_minority"].unique():
                    sub_data = category_data[category_data["category_minority"] == category_minority]

                    # Calculate metrics for the current category and category_minority
                    acc = accuracy_score(sub_data["label"], sub_data["preds"])
                    mcc = matthews_corrcoef(sub_data["label"], sub_data["preds"])
                    f1 = f1_score(sub_data["label"], sub_data["preds"])

                    # Prepare the new row
                    new_row = {
                        "test": self.__class__.__name__,
                        "model": get_model_name(model_checkpoint),
                        "category": category,
                        "category_minority": category_minority,
                        "MCC": mcc,
                        "F1": f1,
                        "Accuracy": acc,
                    }

                    # Check for duplicates
                    if not existing_results.empty and not existing_results[
                        (existing_results["test"] == new_row["test"]) &
                        (existing_results["model"] == new_row["model"]) &
                        (existing_results["category"] == new_row["category"]) &
                        (existing_results["category_minority"] == new_row["category_minority"])
                    ].empty:
                        continue  # Skip saving if the row already exists

                    # Append new row to the results
                    results_store.append(new_row)

        else:
            # Calculate general metrics (for other test types)
            acc = accuracy_score(result["label"], result["preds"])
            mcc = matthews_corrcoef(result["label"], result["preds"])
            f1 = f1_score(result["label"], result["preds"])

            # Prepare the new row
            new_row = {
                "test": self.__class__.__name__,
                "model": get_model_name(model_checkpoint),
                "category": None,
                "category_minority": None,
                "MCC": mcc,
                "F1": f1,
                "Accuracy": acc,
            }

            # Check for duplicates
            if not existing_results.empty and not existing_results[
                (existing_results["test"] == new_row["test"]) &
                (existing_results["model"] == new_row["model"])
            ].empty:
                return  # Skip saving if the row already exists

            # Append new row to the results
            results_store.append(new_row)

        # Save incrementally to the CSV file
        new_data = pd.DataFrame(results_store)
        if not new_data.empty:
            new_data.to_csv(file_path, mode="a", header=not os.path.exists(file_path), index=False)

        print(f"Results saved to {file_path} for {get_model_name(model_checkpoint)}.")
