"""
File that runs everything
1. Read the config
    - define dataset
    - define summarization method (ex: tf-idf, )
    - decide saving path images
    - select a subset of documents
    - save flag (for the images)
    
2. Read entire df -> df with col "text", "mask (0 or 1)"

3. Apply summarizer -> df with col "text", "summary"

4. Image generation:
    input: df with col "text", "summary" (only for rows where mask == 1)
    output:
        save all the images to the folder specified in config
        df with a new column with the path to the image "image_path" 

5. Save the df with "text", "summary", "img_path"
"""