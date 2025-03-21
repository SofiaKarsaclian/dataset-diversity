import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from checklist.DIR import LoadedTest
from checklist.INV import LocationsTest, PrejudiceTest, PronounsTest
from checklist.MFT import FactualTest
from huggingface_hub import HfApi


api = HfApi()
models = api.list_models(
    author= "skarsa",
    #search= "babe"
)

substrings = ["babe", "annomatic"]
model_list = [model for model in models if any(sub in model.id for sub in substrings)]

# Print out the filtered model names
print("Filtered Models:", [model.id for model in model_list])
#model_list = list(models)

test_list = [
    LoadedTest,
    LocationsTest,
    PrejudiceTest,
    PronounsTest,
    FactualTest
]

# for test in test_list:
#     print(f"Running {test.__name__} on all models...")
#     t = test("checklist/data")
#     for model_info in model_list:
#         model_id = model_info.id
#         print(f"running on {model_id} ...")
#         t.execute(model_id)


results_store = []

# Iterate over each test case
for test in test_list:
    print(f"Running {test.__name__} on all models...")
    t = test("checklist/data")

    # Iterate over each model in the model list
    for model_info in model_list:
        model_id = model_info.id

        try:
            print(f"Running on {model_id}...")
            t.execute(model_id)
        except Exception as e:
            print(f"Error while running {test.__name__} on {model_id}: {e}")

print("All results have been saved to 'results.csv'.")
