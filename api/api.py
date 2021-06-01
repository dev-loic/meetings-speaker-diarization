from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from msd.SpeakerAward import SpeakerAward
import shutil
import ffmpeg
import os

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

@app.post("/diarizeMeeting")
def diarize(id, file: UploadFile = File(...)):    
    
    # Convert opus to wav
    with open("test.opus", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with open("output.wav", "wb"):
        stream = ffmpeg.input("test.opus")
        stream = ffmpeg.output(stream, 'output.wav')
        ffmpeg.overwrite_output(stream).run()

    # Diarize
    """ TODO: (Loic Saillant) 2021/06/01 Implement diarizing part asynchronously """
    # diarizer = SpeakerAward("dia")
    # diarizer.apply_diarizer('output.wav')
    
    # Clean
    os.remove('test.opus')
    os.remove('output.wav')
    return { 'Succeed': 'OK' }
