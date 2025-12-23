import os
import torch
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'nsf_hifigan'))
from nvSTFT import STFT
from models import load_model,load_config

from torchaudio.transforms import Resample

def mask_feature(feature1, feature2, mask_rate):
    # 确定要掩码的片段的长度
    length = min(feature1.size(1), feature2.size(1))
    feature1 = feature1[:, :length]
    feature2 = feature2[:, :length]
        
    mask_length = int(length * mask_rate)
    
    # 生成掩码位置的索引
    mask_start = torch.randint(0, length - mask_length, (feature1.size(0),))
    mask_end = mask_start + mask_length
    
    # 创建掩码
    mask = torch.zeros_like(feature1, dtype=torch.bool)
    for i in range(feature1.size(0)):
        mask[i, mask_start[i]:mask_end[i]] = True
    
    # 对第二个特征中对应的掩码片段进行替换
    masked_feature2 = torch.where(mask, feature1, feature2)
    
    return masked_feature2

def mix_feature(feature1, feature2, mix_rate):
    # 确定要混合的片段的长度
    length = min(feature1.shape[1], feature2.shape[1])
    
    # 创建混合的比例
    mix_rate = np.random.uniform(0, mix_rate)
    
    # 对两个特征向量进行混合
    mixed_feature = feature1[:, :length, :] * mix_rate + feature2[:, :length, :] * (1 - mix_rate)
    
    return mixed_feature

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device = device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
            
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()
        
    def extract(self, audio, sample_rate=0, keyshift=0):
                
        # resample
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate) 
            if key_str not in self.resample_kernel: # create new kernel
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device) 
            audio_res = self.resample_kernel[key_str](audio)    
        
        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel
    
    def batch_extract(self, audio, sample_rate=0, keyshift=0): 
        batch_size = audio.size(0)
        print('batch_size: ', batch_size)
        batch_mel = []
        for i in range(batch_size):
            mel = self.extract(audio[i], sample_rate, keyshift)
            batch_mel.append(mel)
        return batch_mel
   
    def infer(self, mel, f0=None):
        # f0 = f0[:,:mel.size(1),0] # B, n_frames
        if f0 is not None:
            f0 = f0[:,:mel.size(1)] # F0 shape: B, n_frames
            audio = self.vocoder(mel, f0)
        else:
            audio = self.vocoder(mel)
        return audio
    
    def batch_infer(self, mel, f0=None):
        batch_size = mel.size(0)
        print('batch_size: ', batch_size)
        batch_audio = []
        if f0 is not None:
            for i in range(batch_size):
                audio = self.infer(mel[i], f0[i])
                batch_audio.append(audio)
        else:
            for i in range(batch_size):
                audio = self.infer(mel[i])
                batch_audio.append(audio)
        return batch_audio
        
        
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.num_mels
        
    def extract(self, audio, keyshift=0):       
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel
    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class NsfHifiGANLog10(NsfHifiGAN):    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio