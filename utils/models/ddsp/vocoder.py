import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from .unit2control import Unit2ControlFacV5A
from .core import upsample
from torch import nn


CREPE_RESAMPLE_KERNEL = {}
F0_KERNEL = {}

class F0_Extractor:
    def __init__(self, f0_extractor, sample_rate = 44100, hop_size = 512, f0_min = 65, f0_max = 800):
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        if f0_extractor == 'crepe':
            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width = 128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        if f0_extractor == 'rmvpe':
            if 'rmvpe' not in F0_KERNEL :
                from rmvpe import RMVPE
                F0_KERNEL['rmvpe'] = RMVPE('pretrain/rmvpe/model.pt', hop_length=160)
            self.rmvpe = F0_KERNEL['rmvpe']
                
    def extract(self, audio, uv_interp = False, device = None, silence_front = 0): # audio: 1d numpy array
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1
                
        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]
            
        # extract f0 using rmvpe
        if self.f0_extractor == "rmvpe":
            f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
            
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")
                    
        # interpolate the unvoiced f0 
        if uv_interp:
            uv = f0 == 0 # unvoiced frames bool, e.g. [True, False, False, True, False, True]
            if len(f0[~uv]) > 0: #  if there are voiced frames
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv]) 
            f0[f0 < self.f0_min] = self.f0_min
        return f0

    def batch_extract(self, audios, uv_interp=False, device=None, silence_front=0):
        processed_f0s = []
        for audio in audios:
            # Extract f0 using rmvpe
            if self.f0_extractor == "rmvpe":
                f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
                f0 = torch.tensor(f0, dtype=torch.float32, device=device)  # Convert to torch tensor
                n_frames = int(len(audio) // self.hop_size) + 1
                start_frame = int(silence_front * self.sample_rate / self.hop_size)
                real_silence_front = start_frame * self.hop_size / self.sample_rate
                audio = audio[int(np.round(real_silence_front * self.sample_rate)):]

                target_time = self.hop_size / self.sample_rate * torch.arange(n_frames - start_frame, device=device)
                f0 = F.interpolate(f0.unsqueeze(0).unsqueeze(0), size=n_frames - start_frame, mode='linear').squeeze()

            else:
                raise ValueError(f"Unknown f0 extractor: {self.f0_extractor}")

            processed_f0s.append(f0)
        
        processed_f0s = torch.stack(processed_f0s, 0) # Convert list of tensors to tensor
        return processed_f0s

class Volume_Extractor:
    def __init__(self, hop_size = 512):
        self.hop_size = hop_size
        
    def extract(self, audio): # audio: 1d numpy array
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = np.array([np.mean(audio2[int(n * self.hop_size) : int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

class CombSubFastFacV5A(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            use_tfm=False,
            pcmer_norm=False,
            mode=None):
        super().__init__()

        print(' [LOAD] HQ-SVC Model ...')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }

        self.unit2ctrl = Unit2ControlFacV5A(
            input_channel=n_unit, 
            output_splits=split_map, 
            use_pitch_aug=use_pitch_aug, 
            pcmer_norm=pcmer_norm
        )
        self.mode = mode
    def forward(self, units_frames, f0_frames, volume_frames, spk, spk_id=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        outputs = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, spk_id, aug_shift=aug_shift, is_infer=infer)

        ctrls, hidden, timbre = outputs
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]
        
        if 'adaln_mlp' in self.mode:
            return signal, hidden, timbre_f0, timbre, style
        else:
            return signal, hidden, timbre