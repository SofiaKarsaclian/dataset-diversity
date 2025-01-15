
import logging
import warnings
import torch
import transformers
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

transformers.logging.set_verbosity(transformers.logging.ERROR)
logging.disable(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define metrics
def compute_metrics_hf(preds):
    """computes F1 score and accuracy over dataset
    Args:
        model (any type): model for evaluation
        testing_dataloader (huggingface dataset): self explained
    Returns:
        dict
    """
    y_pred=preds.predictions
    y_true=preds.label_ids
    y_pred=y_pred.argmax(axis=1)
    mcc=matthews_corrcoef(y_true,y_pred)
    f1=f1_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)
    roc_auc=roc_auc_score(y_true,y_pred)
    return{
        "mcc":mcc,
        "f1":f1,
        "precision":precision,
        "recall":recall,
        "roc_auc":roc_auc,
    }

def compute_metrics(testing_dataloader,model):
    """computes F1 score and accuracy over dataset
    Args:
        model (any type): model for evaluation
        testing_dataloader (huggingface dataset): self explained
    Returns:
        dict
    """
    y_true=[]
    y_pred=[]
    model.eval()
    
    for batch in testing_dataloader:
        batch={k:v.to(model.device)for k,v in batch.items()}

        batch["labels"] = batch["labels"].long()
        with torch.no_grad():
            outputs=model(**batch)
            
        logits=outputs.logits
        predictions=torch.argmax(logits,dim=-1)
        y_true.extend(batch["labels"].type(torch.LongTensor).tolist())  
        y_pred.extend(predictions.type(torch.LongTensor).tolist()) 

    mcc=matthews_corrcoef(y_true,y_pred)
    f1=f1_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)
    roc_auc=roc_auc_score(y_true,y_pred)
    return{
        "mcc":mcc,
        "f1":f1,
        "precision":precision,
        "recall":recall,
        "roc_auc":roc_auc,
    }



