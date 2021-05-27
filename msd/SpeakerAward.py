import json
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import os
from pydub import AudioSegment

class SpeakerAward():
    
    def __init__(self,diarizer):
        self.diarizer = diarizer
        self.json_outputs = None

    def get_json(self):
        load_dotenv()
        self.json_outputs = []
        i = 0
        for segment, _, label in self.diarizer.diarization.itertracks(yield_label=True):
            start = segment.start
            end = start + segment.duration
            speaker = label
            self.json_outputs.append({'speaker':speaker,'start':start,'end':end})
            i+=1
        speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_API_KEY'), region="francecentral", speech_recognition_language = 'en-US')
        for i in range(len(self.json_outputs)):
            t1 = self.json_outputs[i]['start'] * 1000 #Works in milliseconds
            t2 = self.json_outputs[i]['end'] * 1000
            newAudio = AudioSegment.from_wav("data/martin2.wav")
            newAudio = newAudio[t1:t2]
            newAudio.export(f"outputs/batches_videos/temp.wav", format="wav")
            audio_config = speechsdk.audio.AudioConfig(filename=f"outputs/batches_videos/temp.wav")
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = speech_recognizer.recognize_once_async().get()
            self.json_outputs[i]['text']=result.text #Exports to a wav file in the current path.
        os.remove("outputs/batches_videos/temp.wav")
        return json.dumps(self.json_outputs)
