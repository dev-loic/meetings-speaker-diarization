import json
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import audio
from dotenv import load_dotenv
import os
import pydub
import time
from msd.SpeakerDiarizer import SpeakerDiarizer
import torch

class SpeakerAward(SpeakerDiarizer):
    
    load_dotenv()
    
    #Attributes
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_API_KEY'), region="francecentral", speech_recognition_language = 'en-US')
    
    def __init__(self, name_pipe):
        # , pipeline, diarization, der, current_filename
    
        self.json_outputs = None
        super().__init__(name_pipe)
        self.pipeline = torch.hub.load('pyannote/pyannote-audio', self.name_pipe)
        self.diarization = None
        self.der = None
        self.current_filename = None          
        
    def get_json(self):

        self.json_outputs = []
        audio_wave = pydub.AudioSegment.from_wav(self.current_filename)
        for segment, _, label in self.diarization.itertracks(yield_label=True):
            t1 = segment.start * 1000 #Works in milliseconds
            t2 = (segment.start+segment.duration) * 1000
            newAudio = audio_wave[t1:t2]
            newAudio.export(f"temp.wav", format="wav")
            audio_config = speechsdk.audio.AudioConfig(filename="temp.wav")
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            result = speech_recognizer.recognize_once_async().get()
            self.json_outputs.append({'speaker':label,
                                      'start':time.strftime("%H:%M:%S",time.gmtime(segment.start)),
                                      'end':time.strftime("%H:%M:%S",time.gmtime(segment.start+segment.duration)),
                                      'text':result.text})
        os.remove("temp.wav")
        return self.json_outputs
