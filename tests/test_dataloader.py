import pandas as pd
from unittest import TestCase
from src.doc2img.dataloader import get_dataset_poems


class Test_Dataloader(TestCase):
    def test_poems_passes(self):
        df = pd.DataFrame(columns=["text", "topic"])
        path_to_dataset = "./tests/datasets/poems"
        get_dataset_poems(path_to_dataset, df, 100000)
        self.assertTrue(len(df) == 3)
        self.assertTrue(type(df) == pd.DataFrame)

