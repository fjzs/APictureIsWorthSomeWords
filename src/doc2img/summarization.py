# The pupose of this model is to provide summarization functions
# Given a big input of text, the output is a shorter text

import pandas as pd
import yaml

#reading config file
config_file = './config.yaml'
with open(config_file) as cf_file:
    config = yaml.safe_load(cf_file.read())
   
summary_method = config['summary_method']

def get_summary(df, max_tokens):
    # Adding a column to the df called "summary" of type str
    if summary_method == 'tfidf':
        from doc2img.summarization_tfidf import SummarizerPoems
        summarizer = SummarizerPoems(df=df,top_k = max_tokens)
        df['summary'] = summarizer.summary[0:len(df)]
        return df
