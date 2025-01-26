from checklist.test.test import BaseTest
import pandas as pd
from datasets import load_dataset


class FactualTest(BaseTest):

    def __init__(self, data_path):
        super().__init__(data_path)

    def prepare_test_data(self):
        """
        Prepares the test data by retrieving tables from the 'minorities.db' database
        and storing them in a dictionary.
        """

        d1 = load_dataset("notrichardren/easy_qa")["validation"].to_pandas()
        d2 = load_dataset("truthful_qa", "generation")[
            "validation"
        ].to_pandas()

        d1 = (d1["question"] + " " + d1["right_answer"]).tolist()
        d2 = d2["best_answer"].tolist()

        texts = d1 + d2
        self.test_data = pd.DataFrame.from_dict(
            {"text": texts, "label": [0] * len(texts)}
        )

    def test(self, test_data):
        test_data["preds"] = self.make_predictions(data=test_data)

        return test_data
