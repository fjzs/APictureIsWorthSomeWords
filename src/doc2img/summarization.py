# The pupose of this model is to provide summarization functions
# Given a big input of text, the output is a shorter text

import pandas as pd

def get_summary(method_name:str, df:pd.Dataframe, max_tokens:int=77) -> pd.DataFrame:
    # Add a column to the df called "summary" of type str
    pass
