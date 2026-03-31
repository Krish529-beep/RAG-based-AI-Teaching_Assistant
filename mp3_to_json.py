import whisper
import os
import json
from typing import Dict, Any


model = whisper.load_model("large-v2")
audios = os.listdir("Audios")

for audio in audios:
    print(audio)
    number = audio.split('@')[0]
    title = audio.split('@')[1]
    # print(number,title)
    result = model.transcribe(audio=f'Audios/{audio}', language='hi',task='translate',word_timestamps=False)
    
    chunks = []
    
    for segment in result['segments']:
        chunks.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text']
        })
        
    chunks_With_metadata = {"chunks":chunks,"text":result['text'],"language":result['language']}
        
    with open(f'jsons/{audio}.json','w') as f:
        json.dump(chunks_With_metadata,f,indent=4)