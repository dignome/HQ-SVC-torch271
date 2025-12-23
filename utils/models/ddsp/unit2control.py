import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .pcmer import PCmer

# ================= 辅助函数 =================
def split_to_dict(tensor, tensor_splits):
    """将张量分割为字典形式"""
    labels = []
    sizes = []
    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)
    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))

# ================= 核心组件 =================
class FiLM(nn.Module):
    """特征线性调制层"""
    def __init__(self, feature_dim):
        super().__init__()
        self.scale_fc = nn.Linear(feature_dim, feature_dim)
        self.bias_fc = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x, condition):
        scale = self.scale_fc(condition)
        bias = self.bias_fc(condition)
        return x * scale + bias

# ================= 主模型 =================
class Unit2ControlFacV5A(nn.Module):
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        
        # 1. 音色提取器 (film_mlp 使用简单的 MLP 结构)
        self.timbre_extractor = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 256)
        )
            
        # 2. 特征 Embedding (使用 MLP 结构)
        self.f0_embed = self._make_mlp(1, 256)
        self.phase_embed = self._make_mlp(1, 256)
        self.volume_embed = self._make_mlp(1, 256)

        # 3. 融合模块 (4个特征拼接: timbre_f0, style, phase, volume)
        self.fuse_conv = nn.Conv1d(in_channels=256 * 4, out_channels=256, kernel_size=1)
        self.film = FiLM(256)
        
        # 4. 基础卷积栈
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # 5. Transformer 解码器
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        
        self.norm = nn.LayerNorm(256)

        # 6. 输出层
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(nn.Linear(256, self.n_out))

        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None

    def _make_mlp(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, units, f0, phase, volume, spk, spk_id=None, aug_shift=None, is_infer=False):
        # 1. 基础特征提取
        x = self.stack(units.transpose(1, 2)).transpose(1, 2)
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        batch_size = x.shape[0]

        # 2. 说话人与音色解耦
        # film_mlp 模式下默认 use_tfm 为 True 的逻辑
        timbre_embed_raw = self.timbre_extractor(spk).view(batch_size, 1, -1)
        style_embed_raw = (spk.view(batch_size, 1, -1) - timbre_embed_raw)

        # 3. 编码特征
        timbre_f0 = self.f0_embed((1 + f0 / 700).log()) + timbre_embed_raw
        phase_feat = self.phase_embed(phase / np.pi)
        volume_feat = self.volume_embed(volume)

        # 4. 构建 FiLM 调节条件 (拼接 4 个 256 维特征)
        # style_embed 需要在时间维度上扩展
        style_feat = style_embed_raw.expand(-1, n_frame, -1)
        
        condition_style = torch.cat([timbre_f0, style_feat, phase_feat, volume_feat], dim=-1)
        # 通过 1x1 卷积融合维度回 256
        condition_style = self.fuse_conv(condition_style.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 5. 应用 FiLM 调制
        x = self.film(x, condition_style)

        # 6. 后处理与输出
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
        
        return controls, x, timbre_embed_raw