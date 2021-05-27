import torch
from pyannote.database.util import load_rttm
from pyannote.core import Segment, notebook
from pyannote.metrics.diarization import DiarizationErrorRate

class SpeakerDiarizer():
    
    def __init__(self,name_pipe):
        self.pipeline = torch.hub.load('pyannote/pyannote-audio', name_pipe)
        self.diarization = None
        self.der = None

    def apply_diarizer(self,file_name):
        # apply diarization pipeline on your audio file
        self.diarization = self.pipeline({'audio' : file_name})
    
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
