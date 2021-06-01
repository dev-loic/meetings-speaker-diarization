from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from msd.SpeakerAward import SpeakerAward
import shutil
import ffmpeg
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    
    return {
        "greetings" : "This is our root endpoint!"
    }
    

@app.get("/getfile")
def get_file(bucket_name, file_name):
    
    storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(file_name)
    local_file = blob.download_to_filename(file_name)
    
    return local_file
    
@app.get("/diarize")
def diarize(id):
    
    get_file("wagon-data-589-vigouroux", id)
    
    input_file = id
    
    diarizer = SpeakerAward("dia")
    
    diarizer.apply_diarizer(input_file)
    
    json_output = diarizer.get_json()

    return json_output

@app.post("/speakersLabeling")
def label_speakers(id, background_tasks: BackgroundTasks, file: UploadFile = File(...)):    
    
    # Convert opus to wav
    with open('output.opus', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with open('output.wav', 'wb'):
        stream = ffmpeg.input('output.opus')
        stream = ffmpeg.output(stream, 'output.wav')
        ffmpeg.overwrite_output(stream).run()

    # Diarize
    background_tasks.add_task(diarize, id, 'output.wav')
    
    return { 'Succeed': 'OK' }

def diarize(id, file_name):
    print('[START] Diarizer instanciation')
    diarizer = SpeakerAward("dia")
    print('[END] Diarizer instanciation')
    print('[START] Diarize')
    diarizer.apply_diarizer(file_name)
    print('[END] Diarize')
    json = diarizer.get_json()
    print('[START] Saving to the cloud ☁️')
    save_to_cloud(id, json)
    print('[END] Saving to the cloud ☁️')

    # Clean
    os.remove('output.opus')
    os.remove('output.wav')
    
def save_to_cloud(id, jsonData):
    file_name = f'{id}.json'
    
    with open(file_name, 'w') as json_file:
        json_file.write(json.dumps(jsonData))
    
    storage_client = storage.Client()
    bucket = storage_client.bucket('wagon-data-589-vigouroux')
    blob = bucket.blob(f'results/{file_name}')
    blob.upload_from_filename(file_name)
   
    os.remove(file_name)

@app.get("/speakersLabeling")
def get_speakers_labels(id):
    storage_client = storage.Client()
    bucket = storage_client.bucket('wagon-data-589-vigouroux')
    blob = bucket.blob(f'results/{id}.json')
    jsonData = blob.download_as_string(client=None)
    
    return { 'data': json.loads(jsonData) }
