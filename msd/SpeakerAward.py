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
    load_dotenv()
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_API_KEY'), region="francecentral", speech_recognition_language = 'fr-FR')
    
    def __init__(self, name_pipe):
        self.json_outputs = None
        super().__init__(name_pipe)   
        
    def get_json(self):
        self.profils = self.profil_paths
        self.json_outputs = []
        for segment, _, label in self.diarization.itertracks(yield_label=True):
            t1 = segment.start * 1000 #Works in milliseconds
            t2 = (segment.start+segment.duration) * 1000
            newAudio = self.audio_profiles[t1:t2]
            newAudio.export(f"temp.wav", format="wav")
            audio_config = speechsdk.audio.AudioConfig(filename="temp.wav")
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            result = speech_recognizer.recognize_once_async().get()
            self.json_outputs.append({'speaker':label,
                                      'start':time.strftime("%H:%M:%S",time.gmtime(segment.start)),
                                      'end':time.strftime("%H:%M:%S",time.gmtime(segment.start+segment.duration)),
                                      'text':result.text})
        for count,profil in enumerate(self.profils):
            name = profil.split("/")[-1][:-4]
            speaker_letter = self.json_outputs[len(self.profils) - count - 1]['speaker']
            for segment in self.json_outputs:
                if segment['speaker'] == speaker_letter :
                    segment['speaker'] = name
        for i in range(len(self.profils)):
            del self.json_outputs[i]
        os.remove("temp.wav")
        return self.json_outputs
