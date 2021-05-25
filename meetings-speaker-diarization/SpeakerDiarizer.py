import torch
import torchaudio

class SpeakerDiarizer:
  wav2mel = torch.jit.load("wav2mel.pt")
  dvector = torch.jit.load("dvector-step250000.pt").eval()