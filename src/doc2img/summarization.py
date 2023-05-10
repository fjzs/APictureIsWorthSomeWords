# The pupose of this model is to provide summarization functions
# Given a big input of text, the output is a shorter text

from transformers import pipeline
from doc2img.summarization_hf import *
from doc2img.summarization_tfidf import *

def get_summary(df, config):
    if config['summary_method'] == 'tf-idf':
        summarizer = SummarizerPoems(100)
        df['summary'] = summarizer.summary[0:len(df)]
        return df
    
    elif config['summary_method']=='hf':
        return text_summarization_hf(df, config)
    