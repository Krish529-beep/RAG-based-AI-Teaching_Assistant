import joblib 
import requests
import joblib
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

def inference(prompt):
    r = requests.post('http://localhost:11434/api/generate', json={
        # "model": "deepseek-r1",
        'model':'llama3.2',
        "prompt": prompt,
        "stream": False,
    })
    
    res = r.json()
    return res


df = joblib.load('embeddings.joblib')


# Query input
incoming_query = input("\nAsk your question: ")

query_embedding = create_embedding([incoming_query])[0]


# Compute similarity
similarities = cosine_similarity(
    np.vstack(df['embedding']),
    [query_embedding]
).flatten()


# Get top 5 matches
top_k = 5
max_idx = np.argsort(similarities)[-top_k:][::-1]


# Fetch results
new_df = df.loc[max_idx]


prompt = f'''Hithesh sir is teaching django development in chai aur django course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "video_number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.
Dont use hardcoed seconds in your answer.
convert seconds in to minutes and seconds format in your answer.
also use video 1 for such 01@1 video number and so on. Dont mention the video number as 01, 02, 03, instead mention it as video 1, video 2, video 3 and so on. Dont mention the above format in your answer, its just for you to understand the context. Answer in a human way.
'''

with open('prompt.txt', 'w') as f: 
    f.write(prompt)

print(inference(prompt)['response'])

