import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d

# 外部依赖导入
from .diffusion import GaussianDiffusion, CFGDiffusion
from .wavenet import WaveNet, ControlWaveNet
from .ddsp.vocoder import CombSubFastFacV5A

# ================= 辅助函数 =================

def infonce_loss(spk_embeddings, spk_ids, temperature=0.1, supervised=True):
    """计算音频特征的对比损失"""
    spk_embeddings = F.normalize(spk_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(spk_embeddings, spk_embeddings.T) / temperature
    
    if supervised:
        mask = (spk_ids.unsqueeze(1) == spk_ids.t().unsqueeze(0)).float()
        pos_mask = mask - torch.diag(torch.ones(mask.shape[0], device=mask.device))
        neg_mask = 1 - mask
    else:
        pos_mask = torch.eye(spk_embeddings.size(0), device=spk_embeddings.device).bool()
        neg_mask = ~pos_mask
    
    pos_mask_add = neg_mask * (-1000)
    neg_mask_add = pos_mask * (-1000)
    log_infonce_per_example = (similarity_matrix * pos_mask + pos_mask_add).logsumexp(-1) - \
                              (similarity_matrix * neg_mask + neg_mask_add).logsumexp(-1)
    return -torch.mean(log_infonce_per_example)

def get_f0_loss(spk, f0_pred_mu, f0_pred_var, f0_gt):
    """计算 F0 预测的 L1 损失"""
    f0_gt = torch.log(1 + f0_gt / 700)
    f0_gt_mu = f0_gt.mean(dim=1)
    f0_gt_var = f0_gt.var(dim=1)
    loss = F.l1_loss(f0_pred_mu, f0_gt_mu) + F.l1_loss(f0_pred_var, f0_gt_var)
    return loss

def adjust_f0(src_f0, src_log_f0_mean, src_log_f0_var, tar_log_f0_mean, tar_log_f0_var):
    """根据目标分布调整 F0"""
    semitone_difference = 12 * (tar_log_f0_mean - src_log_f0_mean) / torch.log(torch.tensor(2.0))
    semitone_difference_rounded = torch.round(semitone_difference)
    adjustment_factor = torch.pow(2, semitone_difference_rounded / 12)
    adjusted_f0 = src_f0 * adjustment_factor
    return adjusted_f0, semitone_difference_rounded

# ================= 核心组件类 =================

class SpeakerClassifier(nn.Module):
    def __init__(self, input_dim=256, num_speakers=100):
        super(SpeakerClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_speakers)

    def forward(self, x):
        logits = self.fc(x)
        prob = F.softmax(logits, dim=-1)
        pred_label = torch.argmax(prob, dim=-1)
        return logits, pred_label

class F0Predictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1):
        super(F0Predictor, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_branch = nn.Linear(hidden_dim, output_dim)
        self.var_branch = nn.Linear(hidden_dim, output_dim)

    def forward(self, spk):
        features = self.shared_layers(spk)
        f0_mean = self.mean_branch(features)
        f0_var = self.var_branch(features)
        return f0_mean, f0_var

# ================= 主模型类 =================

class HQ_SVC(torch.nn.Module):
    def __init__(self, hop_size, args):
        super(HQ_SVC, self).__init__()
        self.hop_size = 512
        in_channels = 256
        num_mels = 128
        self.sampling_rate = 44100
        n_unit = 256
        pcmer_norm = False
        out_dims = 128
        n_layers = 20
        n_chans = 512
        n_hidden = 256

        self.guidance_scale = args.guidance_scale
        self.drop_rate = args.drop_rate
        self.use_pitch_aug = False
        self.use_tfm = args.use_tfm
        self.mode = args.mode
        self.use_mi_loss = args.use_mi_loss
        self.use_style_loss = args.use_style_loss
        self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)

        if self.guidance_scale is not None and self.guidance_scale >= 0:
            self.ddsp_model = CombSubFastFacV5A(self.sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm, mode=self.mode)
            self.diff_model = CFGDiffusion(ControlWaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims=out_dims, drop_rate=self.drop_rate)
        else:
            self.ddsp_model = CombSubFastFacV5A(self.sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm, mode=self.mode)
            self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden))

        self.speaker_classifier = SpeakerClassifier(input_dim=256, num_speakers=100)
        self.ce = nn.CrossEntropyLoss()
        if 'pred_f0' in self.mode:
            self.f0_predictor = F0Predictor(input_dim=256, hidden_dim=512, output_dim=1)

    def forward(self, x, f0, volume, spk, spk_id=None, src_spk=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift=False, vocoder=None, return_wav=False, use_ssim_loss=False, return_timbre=False):
        device = x.device
        if 'pred_f0' in self.mode and src_spk is not None and infer:
            tar_log_f0_mean, tar_log_f0_var = self.f0_predictor(self.ddsp_model.unit2ctrl.timbre_extractor(spk))
            src_log_f0_mean, src_log_f0_var = self.f0_predictor(self.ddsp_model.unit2ctrl.timbre_extractor(src_spk))
            f0, shift_key = adjust_f0(f0, src_log_f0_mean, src_log_f0_var, tar_log_f0_mean, tar_log_f0_var)
            print(f'shift key: {shift_key}')
        
        outputs = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        if 'adaln_mlp' in self.mode:
            ddsp_wav, hidden, timbre_f0, timbre, style = outputs
        else:
            ddsp_wav, hidden, timbre = outputs

        if return_timbre:
            return timbre
        
        ddsp_mel = vocoder.extract(ddsp_wav) if vocoder is not None else None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)

        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            if self.guidance_scale is not None and self.guidance_scale >= 0:
                diff_loss = self.diff_model(hidden, timbre_f0, gt_spec=gt_spec, k_step=k_step, infer=False, guidance_scale=self.guidance_scale)
            else:
                diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            
            spk_loss = infonce_loss(timbre, spk_id.to(device), 0.1) if (spk_id is not None and 'infonce' in self.mode) else torch.tensor(0.).to(device)
            f0_loss = get_f0_loss(spk, *self.f0_predictor(timbre), f0.unsqueeze(-1)) if 'pred_f0' in self.mode else torch.tensor(0.).to(device)
            mi_loss = torch.tensor(0.).to(device)
            style_loss = torch.tensor(0.).to(device)

            return ddsp_loss, diff_loss, spk_loss, mi_loss, style_loss, f0_loss
        else:
            if gt_spec is not None:
                b, t, d = ddsp_mel.shape
                ddsp_mel = gt_spec[:, :t, :d]
            
            if k_step > 0:
                if self.guidance_scale is not None and self.guidance_scale >= 0:
                    mel = self.diff_model(hidden, timbre_f0, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm, guidance_scale=self.guidance_scale)
                else:
                    mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            
            return vocoder.infer(mel, f0) if return_wav else mel

# ================= 入口加载函数 =================

def load_hq_svc(mode='train', model_path=None, device='cuda', hop_size=512, args=None):
    generator = HQ_SVC(hop_size, args).to(device)
    if mode in ['infer', 'finetune']:
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict, strict=False)
        if mode == 'infer':
            generator.eval()
    else:
        generator.train()
    return generator