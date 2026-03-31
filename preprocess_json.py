import joblib 
import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    return r.json()['embeddings']


# Load all JSON files
json_files = os.listdir("jsons")

my_dicts = []
chunk_id = 0

for json_file in json_files:
    number = json_file.split('_')[0]  # extract video_number

    with open(f'jsons/{json_file}', 'r') as f:
        content = json.load(f)

        texts = [chunk['text'] for chunk in content['chunks']]
        embeddings = create_embedding(texts)

        print(f'Creating embeddings for {json_file}')

        for i, chunk in enumerate(content['chunks']):
            chunk['embedding'] = embeddings[i]
            chunk['chunk_id'] = chunk_id
            chunk['video_number'] = number

            chunk_id += 1
            my_dicts.append(chunk)


# Create DataFrame
df = pd.DataFrame.from_records(my_dicts)

# save the dataframe
joblib.dump(df,'embeddings.joblib')