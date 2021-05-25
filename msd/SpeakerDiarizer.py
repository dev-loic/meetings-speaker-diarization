import torch
import torchaudio
from spectralcluster import SpectralClusterer
import numpy as np
import os

class SpeakerDiarizer:
  
  def __init__(self):
    this_dir, _ = os.path.split(__file__)
    wav2mel_model_path = os.path.join(this_dir, "data", "wav2mel.pt")
    self.wav2mel = torch.jit.load(wav2mel_model_path)
    dvector_model_path = os.path.join(this_dir, "data", "dvector-step250000.pt")
    self.dvector = torch.jit.load(dvector_model_path).eval()
    self.sample_rate = None
    self.wav_tensor = None
    self.emb_tensor = None
    self.speaker_label = None
  
  def load(self, audio_file_path):
    self.wav_tensor, self.sample_rate = torchaudio.load(audio_file_path)
  
  def generate_dvectors(self):
    # nb_windows = audio without silence / frame_rate
    mel_tensor = self.wav2mel(self.wav_tensor, self.sample_rate)  # shape: (nb_windows, n_mels)
    emb_tensor = self.dvector.embed_utterances(mel_tensor)  # shape: (emb_dim)
    
    self.emb_tensor = emb_tensor
    
  def spectral_clustering(self, min_clusters=1, max_clusters = 100, p_percentile=0.95, gaussian_blur_sigma=1):
    
    """
    Convert the embedded the d vector from segmentation to a similarity matrix that ouput labels for each segments   
    """
    clusterer = SpectralClusterer(min_clusters,
                                  max_clusters,
                                  p_percentile,
                                  gaussian_blur_sigma)
    embedded_input = []
    
    for i in range(len(self.emb_tensor)):

      embedded_input.append(self.emb_tensor[i].detach().numpy())
      
    embedded_input = np.array(embedded_input)
    
    speaker_label = clusterer.predict(embedded_input)
    
    self.speaker_label = speaker_label