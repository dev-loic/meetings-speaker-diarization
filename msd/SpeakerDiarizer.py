import torch
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
import pydub
import os

class SpeakerDiarizer():
    
    #Attributes
    testing_paths=["data/Martin.wav","data/Loïc.wav","data/Yoann.wav"]
    
    def __init__(self,name_pipe):
        self.name_pipe = name_pipe
        self.pipeline = torch.hub.load('pyannote/pyannote-audio', self.name_pipe)
        self.diarization = None
        self.der = None
        self.current_filename = None
        self.audio_profiles = None
        self.profil_paths = None

    def apply_diarizer(self, file_name, profil_paths = testing_paths):
        # apply diarization pipeline on your audio file
        self.current_filename = pydub.AudioSegment.from_wav(file_name)
        if self.current_filename.duration_seconds < 180:
            self.audio_profiles = self.current_filename * int(180/self.current_filename.duration_seconds+1)
        else :
            self.audio_profiles = self.current_filename
        self.profil_paths = profil_paths
        for profil in profil_paths:
            profil_wav = pydub.AudioSegment.from_wav(profil)    
            self.audio_profiles = profil_wav[:] + self.audio_profiles[:]
        self.audio_profiles.export("avecprofil.wav", format="wav")
        self.diarization = self.pipeline({'audio' : "avecprofil.wav"})
        os.remove("avecprofil.wav")
    
    def write_rttm(self,path_outputs):
        with open(path_outputs, 'w') as f:
            self.diarization.write_rttm(f)
    
    def print_outputs(self):
        for turn, _, speaker in self.diarization.itertracks(yield_label=True):
            print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')
            
    def score(self,path_reference):
        reference = load_rttm(path_reference)
        reference = reference[list(reference.keys())[0]]
        metric = DiarizationErrorRate()
        self.der = metric(reference=reference, hypothesis=self.diarization,detailed=True)
        print(self.der)
