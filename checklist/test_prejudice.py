import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from checklist.INV import PrejudiceTest
from huggingface_hub import HfApi


api = HfApi()
models = api.list_models(
    author= "skarsa",
    search= "annomatic"
)

model_list = list(models)


results_store = []


print(f"Running Prejudice Test on all models...")
t = PrejudiceTest("checklist/data")

# Iterate over each model in the model list
for model_info in model_list:
    model_id = model_info.id

    try:
        print(f"Running on {model_id}...")
        t.execute(model_id, results_store=results_store)
    except Exception as e:
        print(f"Error while running test on {model_id}: {e}")

print("All results have been saved to 'results.csv'.")
