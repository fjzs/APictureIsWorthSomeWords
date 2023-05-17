from unittest import TestCase
from src.doc2img.summarization_tfidf import SummarizerTFIDF
from src.doc2img.dataloader import get_raw_dataset

class Test_Summarizer_TFID(TestCase):
    
    def test_tfidf_poems(self):
        df = get_raw_dataset(dataset_name="poems")
        summarizer = SummarizerTFIDF(df)
        for i in range(10):
            summary_i = summarizer.get_summary_of_index(i)
            self.assertTrue(type(summary_i) == str)
    
    def test_tfidf_nyt(self):
        df = get_raw_dataset(dataset_name="nyt")
        summarizer = SummarizerTFIDF(df)
        for i in range(10):
            summary_i = summarizer.get_summary_of_index(i)
            self.assertTrue(type(summary_i) == str)
        
        
   