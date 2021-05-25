import torch
import torchaudio

class SpeakerDiarizer:
  
  # Attributes
  wav2mel = torch.jit.load("data/pretrained_models/wav2mel.pt")
  dvector = torch.jit.load("data/pretrained_models/dvector-step250000.pt").eval()
  
  def __init__(self):
    self.sample_rate = None
    self.wav_tensor = None
    self.emb_tensor = None
  
  def load(self, audio_file_path):
    self.wav_tensor, self.sample_rate = torchaudio.load(audio_file_path)
  
  def generate_dvectors(self):
    # nb_windows = audio without silence / frame_rate
    mel_tensor = self.wav2mel(self.wav_tensor, self.sample_rate)  # shape: (nb_windows, n_mels)
    emb_tensor = self.dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
    
    self.emb_tensor = emb_tensor
