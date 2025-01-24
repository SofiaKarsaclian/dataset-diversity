import sys
sys.path.append('C:/Users/sofik/Documents/Uni Konstanz/SEDS/Thesis/master-thesis')
import pandas as pd
import os
from checklist.DIR import LoadedTest
from checklist.INV import LocationsTest, PrejudiceTest, PronounsTest
from checklist.MFT import FactualTest
from huggingface_hub import HfApi


api = HfApi()
models = api.list_models(
    author= "skarsa",
    search= "annomatic"
)

model_list = list(models)

test_list = [
    LoadedTest,
    # LocationsTest,
    # PrejudiceTest,
    # PronounsTest,
    # FactualTest
]

# for test in test_list:
#     print(f"Running {test.__name__} on all models...")
#     t = test("checklist/data")
#     for model_info in model_list:
#         model_id = model_info.id
#         print(f"running on {model_id} ...")
#         t.execute(model_id)

selected_model = model_list[1]

results_store = []

for test in test_list:
    print(f"Running {test.__name__} on all models...")
    t = test("checklist/data")
    
    # Iterate over each model in the model list
    for model_info in model_list:
        model_id = model_info.id
        print(f"Running on {model_id} ...")
        
        # Execute the test and append the results to results_store
        t.execute(model_id, results_store=results_store)
        
        # Convert the results to a DataFrame
        result_df = pd.DataFrame(results_store)
        
        # Append the results incrementally to the CSV file
        file_path = "checklist/data/results.csv"
        if os.path.exists(file_path):
            result_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            result_df.to_csv(file_path, mode='w', header=True, index=False)

        print(f"Results saved to {file_path} for {model_id}.")

print("All results have been saved incrementally to 'results.csv'.")

