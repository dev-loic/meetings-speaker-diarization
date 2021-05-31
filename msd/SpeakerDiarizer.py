import torch
from pyannote.database.util import load_rttm
from pyannote.core import Segment, notebook
from pyannote.metrics.diarization import DiarizationErrorRate
import pydub

class SpeakerDiarizer():
    
    def __init__(self,name_pipe):
        self.name_pipe = name_pipe
        self.pipeline = torch.hub.load('pyannote/pyannote-audio', self.name_pipe)
        self.diarization = None
        self.der = None
        self.current_filename = None
        self.audio_profiles = None
        self.profil_paths = None

    def apply_diarizer(self,file_name,profil_paths):
        # apply diarization pipeline on your audio file
        self.current_filename = file_name
        self.profil_paths = profil_paths
        self.audio_profiles = pydub.AudioSegment.from_wav(self.current_filename)
        for profil in profil_paths:
            profil_wav = pydub.AudioSegment.from_wav(profil)    
            self.audio_profiles = profil_wav[:] + self.audio_profiles[:]
        self.audio_profiles.export("avecprofil.wav", format="wav")
        self.diarization = self.pipeline({'audio' : "avecprofil.wav"})
    
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
