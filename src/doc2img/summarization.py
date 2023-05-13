# The pupose of this model is to provide summarization functions
# Given a big input of text, the output is a shorter text

import pandas as pd

# #reading config file
# config_file = './config.yaml'
# with open(config_file) as cf_file:
#     config = yaml.safe_load(cf_file.read())
   

from doc2img.summarization_hf import *
from doc2img.summarization_tfidf import *

def get_summary(df, config):
    method = config['summary_method']
    if method == 'tfidf':
        from doc2img.summarization_tfidf import SummarizerPoems
        summarizer = SummarizerPoems(df=df,top_k = config['summarization']['max_tokens'])
        df['summary'] = summarizer.summary[0:len(df)]
        return df
    
    elif method == 'hf':
        return text_summarization_hf(df, config)
    
    else:
        raise ValueError(f"Method {method} not implemented")