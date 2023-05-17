# The pupose of this model is to provide summarization functions
# Given a big input of text, the output is a shorter text

import pandas as pd

from doc2img.summarization_hf import *
from doc2img.summarization_tfidf import SummarizerTFIDF

def get_summary(df_train: pd.DataFrame, df_test: pd.DataFrame, config: dict):
    """Obtains the summary for the df_test

    Args:
        df_train (pd.DataFrame):
        df_test (pd.DataFrame):
        config (dict):

    Raises:
        ValueError: _description_

    Returns:
        df_test: with the "summary" column
    """

    method = config['summary_method']
    max_tokens = config['summarization']['max_tokens']

    if method == 'tfidf':        
        summarizer = SummarizerTFIDF(df=df_train, top_k = max_tokens)
        return summarizer.get_summary_of_dftest(df_test)
    
    elif method == 'hf':
        return text_summarization_hf(df_test, config)
    
    else:
        raise ValueError(f"Method {method} not implemented")