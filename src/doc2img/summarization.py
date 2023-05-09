# The pupose of this model is to provide summarization functions
# Given a big input of text, the output is a shorter text

from transformers import pipeline

def get_summary(config, df):
    if config['default']['summary_method'] == 'tf-idf':
        pass
    elif config['default']['summary_method']=='hf':
        return text_summarization_hf(config, df)


def text_summarization_hf(config, df):
    summarizer =  pipeline("summarization", model=config['summarization']['model'],
                            min_length=config['summarization']['min_length'], max_length=config['summarization']['max_length'])
    
    def summarize(text, summarizer):
        # We can add some preprocessing for text here 
        text = text.replace("\n", ". ")
        ans = summarizer(text)
        return ans[0]['summary_text']

    df['summarized_text'] = df['text'].apply(lambda x: summarize(x, summarizer))
    return df
    