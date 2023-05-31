import pandas as pd
import numpy as np
import json

# First Analysis: For a given method on a particular dataset,
# which prompt gives the best CLIP score

# Second Analysis: For a given method on a particular dataset,
# what is the average Median CLIP score from the 
# best performing prompts on each image.
def quant_analysis(df, prompt_list):
    best_prompts = []
    best_scores = []
    prompt_dict = {
        "prompt_{}_score".format(idx): prompt for idx, prompt in enumerate(prompt_list)
    }
    for index, row in df.iterrows():
        best_prompt = None
        curr_best = float("-inf")
        for prompt_score, prompt in prompt_dict.items():
            # Currently only doing this analysis on Median CLIP Score 
            # Which is the first score in the tuple.
            scores = json.loads(row[prompt_score])
            if scores[2] > curr_best:
                curr_best = scores[2]
                best_prompt = prompt
        best_prompts.append(best_prompt)
        best_scores.append(curr_best)
    
    # Plotting the distribution of best prompts
    df['best_prompt'] = best_prompts
    df['best_prompt'].value_counts().plot.barh()

    # Getting Average Median CLIP Score
    df['best_clip_median'] = best_scores
    print("Average Median CLIP Score: {}".format(np.average(best_scores)))

    return df



