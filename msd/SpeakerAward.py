import json
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import audio
from dotenv import load_dotenv
import os
import pydub
import time

class SpeakerAward():
    
    #Attributes
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_API_KEY'), region="francecentral", speech_recognition_language = 'en-US')
    
    def __init__(self,diarizer):
        self.diarizer = diarizer
        self.json_outputs = None

    def get_json(self):
        load_dotenv()
        self.json_outputs = []
        audio_wave = pydub.AudioSegment.from_wav(self.diarizer.current_filename)
        for segment, _, label in self.diarizer.diarization.itertracks(yield_label=True):
            t1 = segment.start * 1000 #Works in milliseconds
            t2 = (segment.start+segment.duration) * 1000
            newAudio = audio_wave[t1:t2]
            newAudio.export(f"temp.wav", format="wav")
            audio_config = speechsdk.audio.AudioConfig(filename=f"temp.wav")
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            result = speech_recognizer.recognize_once_async().get()
            self.json_outputs.append({'speaker':label,
                                      'start':time.strftime("%H:%M:%S",time.gmtime(segment.start)),
                                      'end':time.strftime("%H:%M:%S",time.gmtime(segment.start+segment.duration)),
                                      'text':result.text})
        os.remove("temp.wav")
        return self.json_outputs
