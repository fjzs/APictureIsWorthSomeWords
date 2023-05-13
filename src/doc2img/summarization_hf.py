from transformers import pipeline

def summarize(text, summarizer):
        # We can add some preprocessing for text here 
        text = text.replace("\n", ". ")
        ans = summarizer(text)
        return ans[0]['summary_text']

def text_summarization_hf(df, config):
    summarizer =  pipeline("summarization", model=config['summarization']['model'],
                            min_length=config['summarization']['min_length'], max_length=config['summarization']['max_length'])
#     if config['max_docs']:
#         df = df.sample(n=config['max_docs'])
#         df = df.reset_index(drop=True)
    df['summary'] = df['text'].apply(lambda x: summarize(x, summarizer))
    return df