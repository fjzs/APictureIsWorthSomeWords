# The pupose of this model is to provide summarization functions
# Given a big input of text, the output is a shorter text

from transformers import pipeline
import configparser
from utils.conf import *

config = Conf('utils/conf.txt')

def text_summarization_hf(text, config):
    summarizer =  pipeline("summarization", model=config.summarization['model'],
                            min_length=config.summarization['min_length'], max_length=config.summarization['max_length'])
    # We can add some preprocessing for text here 
    text = text.replace("\n", " ")
    ans = summarizer(text)
    return ans[0]['summary_text']
    