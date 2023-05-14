import pandas as pd
from unittest import TestCase
from src.doc2img.dataloader import get_raw_dataset


class Test_Dataloader(TestCase):
    def test_poems_passes(self):
        expected_columns = ["text", "topic"]
        df = get_raw_dataset("poems")        
        self.assertTrue(type(df) == pd.DataFrame)        
        for c in df.columns:
            self.assertTrue(c in expected_columns)
        
    def test_nyt_passes(self):
        expected_columns = ["text", "topic"]
        df = get_raw_dataset("nyt")        
        self.assertTrue(type(df) == pd.DataFrame)        
        for c in df.columns:
            self.assertTrue(c in expected_columns)