import whisper 
import os 
import subprocess

files = os.listdir("videos")
for file in files:
    # print(file)
    filename = file.split('.')[0]
    print(filename)
    subprocess.run(['ffmpeg','-i',f'Videos/{file}',f'Audios/{filename}.mp3'])