import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import os
import time
from msd.SpeakerDiarizer import SpeakerDiarizer

class SpeakerAward(SpeakerDiarizer):
    
    #Attributes
    load_dotenv()
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_API_KEY'), region="francecentral", speech_recognition_language = 'fr-FR')
    
    def __init__(self, name_pipe):
        self.json_outputs = None
        super().__init__(name_pipe)
        self.time_to_sub = 0
        
    def get_json(self):
        self.json_outputs = []
        for count, (segment, _, label) in enumerate(self.diarization.itertracks(yield_label=True)):
            if count == len(self.profil_paths)-1:
                self.time_to_sub = segment.end - 0.0000001
            if segment.start - self.time_to_sub > self.current_filename.duration_seconds :
                break
            t1 = segment.start * 1000 #Works in milliseconds
            t2 = (segment.start+segment.duration) * 1000
            newAudio = self.audio_profiles[t1:t2]
            newAudio.export(f"temp.wav", format="wav")
            audio_config = speechsdk.audio.AudioConfig(filename="temp.wav")
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            result = speech_recognizer.recognize_once_async().get()
            if result.text == "":
                continue
            self.json_outputs.append({'speaker':label,
                                      'start': time.strftime("%H:%M:%S",time.gmtime(t1/1000-self.time_to_sub)),
                                      'end': time.strftime("%H:%M:%S",time.gmtime(t2/1000-self.time_to_sub)),
                                      'text':result.text})
        for count,profil in enumerate(self.profil_paths):
            name = profil.split("/")[-1][:-4]
            speaker_letter = self.json_outputs[len(self.profil_paths) - count - 1]['speaker']
            for segment in self.json_outputs[len(self.profil_paths):]:
                if segment['speaker'] == speaker_letter :
                    segment['speaker'] = name
        os.remove("temp.wav")
        return self.json_outputs[len(self.profil_paths):]
