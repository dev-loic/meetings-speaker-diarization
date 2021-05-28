from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pytz
from msd.apitest import test
from google.cloud import storage
from msd.SpeakerDiarizer import SpeakerDiarizer
from msd.SpeakerAward import SpeakerAward
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