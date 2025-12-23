import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Mish


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip

class ResidualBlockAdaIN(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.singer_conditioner_projection = nn.Linear(encoder_hidden, 2 * residual_channels)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        # AdaIN layers
        self.adain_scale = nn.Sequential(
            nn.Linear(2 * residual_channels, 2 * residual_channels),
            nn.SiLU(),
            nn.Linear(2 * residual_channels, 2 * residual_channels)
        )
        
        self.adain_shift = nn.Sequential(
            nn.Linear(2 * residual_channels, 2 * residual_channels),
            nn.SiLU(),
            nn.Linear(2 * residual_channels, 2 * residual_channels)
        )
        
        # self.instance_norm = nn.InstanceNorm1d(2 * residual_channels, affine=False)

    def forward(self, x, conditioner, singer_conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        singer_conditioner = self.singer_conditioner_projection(singer_conditioner)
        
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        # Apply AdaIN
        scale = self.adain_scale(singer_conditioner).unsqueeze(-1)  # Shape (B, 2 * residual_channels, 1)
        shift = self.adain_shift(singer_conditioner).unsqueeze(-1)  # Shape (B, 2 * residual_channels, 1)

        # Instance normalization followed by scaling and shifting
        # y = self.instance_norm(y)  # Normalize across the feature dimension for each instance
        
        mean = y.mean(dim=(2), keepdim=True)  # (batch_size, 2 *channels, 1)
        std = y.std(dim=(2), keepdim=True)    # (batch_size, 2 *channels, 1)

        # 执行实例归一化
        y_norm = (y - mean) / (std + 1e-5)  # 防止除以零
        y = y_norm * scale + shift  # Apply AdaIN scaling and shifting

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip

class ResidualBlockFiLM(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.singer_conditioner_projection = nn.Linear(encoder_hidden, 2 * residual_channels)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        
        # FiLM layers
        # self.film_scale = nn.Linear(2 * residual_channels, 2 * residual_channels)
        # self.film_shift = nn.Linear(2 * residual_channels, 2 * residual_channels)
        
        self.adaln_scale = nn.Sequential(
            nn.Linear(2 * residual_channels, 2 * residual_channels),
            nn.SiLU(),
            nn.Linear(2 * residual_channels, 2 * residual_channels)
        )
        
        self.adaln_shift = nn.Sequential(
            nn.Linear(2 * residual_channels, 2 * residual_channels),
            nn.SiLU(),
            nn.Linear(2 * residual_channels, 2 * residual_channels)
        )
        self.layer_norm = nn.LayerNorm(2 * residual_channels)

    def forward(self, x, conditioner, singer_conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        singer_conditioner = self.singer_conditioner_projection(singer_conditioner)
        
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        
        # Apply FiLM/AdaLN
        scale = self.adaln_scale(singer_conditioner)
        shift = self.adaln_shift(singer_conditioner)
        # y = self.layer_norm(y.permute(0, 2, 1) * scale + shift).permute(0, 2, 1)
        
        y = self.layer_norm(y.permute(0, 2, 1))  # 对最后一维（特征维度）进行归一化
        y = (y * scale + shift).permute(0, 2, 1)  # 在归一化后的基础上进行动态调整

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip
    
class ResidualBlockNew(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.audio_conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.singer_conditioner_projection = nn.Linear(encoder_hidden, 2 * residual_channels)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        
        # FiLM 调制层
        self.film_scale = nn.Linear(2 * residual_channels, 2 * residual_channels)
        self.film_shift = nn.Linear(2 * residual_channels, 2 * residual_channels)

    def forward(self, x, audio_conditioner, singer_conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        
        # 处理音频条件
        audio_cond = self.audio_conditioner_projection(audio_conditioner)
        
        # 处理文本条件（假设文本条件的形状为 [batch_size, encoder_hidden]）
        singer_cond = self.singer_conditioner_projection(singer_conditioner.squeeze(-1))  # [batch_size, residual_channels]
        
        # 使用 FiLM 调制来融合文本条件
        scale = self.film_scale(singer_cond).unsqueeze(2)  # [batch_size, residual_channels, 1]
        shift = self.film_shift(singer_cond).unsqueeze(2)  # [batch_size, residual_channels, 1]
        
        # 应用 FiLM 调制到音频条件
        conditioner = (audio_cond * scale + shift)
        
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip
    
# class MultiConditionResidualBlock(nn.Module):
#     def __init__(self, encoder_hidden, residual_channels, dilation):
#         super().__init__()
#         self.residual_channels = residual_channels
#         self.dilated_conv = nn.Conv1d(
#             residual_channels,
#             2 * residual_channels,
#             kernel_size=3,
#             padding=dilation,
#             dilation=dilation
#         )
#         self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
#         self.audio_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
#         self.text_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
#         self.text_dafualt_embed = nn.Embedding(1, residual_channels)
#         self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
    
#     def forward(self, x, audio_condition, text_condition, diffusion_step):
#         diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
#         audio_condition = self.conditioner_projection(audio_condition)
#         if text_condition is not None:
#             text_condition = self.conditioner_projection(text_condition)
#         else:
#             text_condition = self.text_dafualt_embed(torch.zeros(1, dtype=torch.long, device=x.device))
            
#         condition = self.ca(audio_condition, text_condition)
#         # condition = audio_condition + text_condition
#         y = x + diffusion_step

#         y = self.dilated_conv(y) + condition

#         # Using torch.split instead of torch.chunk to avoid using onnx::Slice
#         gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
#         y = torch.sigmoid(gate) * torch.tanh(filter)

#         y = self.output_projection(y)

#         # Using torch.split instead of torch.chunk to avoid using onnx::Slice
#         residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
#         return (x + residual) / math.sqrt(2.0), skip
    

class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=1
            )
            for i in range(n_layers)
        ])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)
        self.n_chans = n_chans
        self.n_hidden = n_hidden
    def forward(self, spec, diffusion_step, audio_cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param audio_cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]
        x = F.relu(x)
        
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, audio_cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x[:, None, :, :]

class TextControlWaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.text_dafualt_embed = nn.Embedding(1, n_hidden) 
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.ca = CrossAttention(n_hidden)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=1
            )
            for i in range(n_layers)
        ])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)
        
        # 映射层用于调制因子
        self.audio_modulation = nn.Linear(n_hidden, n_hidden)
        self.text_modulation = nn.Linear(n_hidden, n_hidden)
        
    def forward(self, spec, diffusion_step, audio_cond, text_cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param audio_cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []

        # # ca
        # if text_cond is None:
        #     # # 生成默认的text_cond，全0张量嵌入
        #     # text_cond = self.text_dafualt_embed(torch.zeros(1, dtype=torch.long, device=x.device))
        #     # text_cond = text_cond.repeat(audio_cond.shape[0], audio_cond.shape[-1], 1)
        #     # text_cond = text_cond.permute(0, 2, 1) # (1, 256, seq_len)
            
        #     # 生成高斯噪声，假设 text_cond 的维度为 (batch_size, seq_length, embed_dim)
        #     text_cond = torch.randn(audio_cond.shape, device=x.device)  # 生成高斯噪声
        # else:
        #     # text_cond = text_cond.repeat(1, 1, audio_cond.shape[-1])
        #     text_cond = text_cond.expand(-1, -1, audio_cond.shape[-1])
        # # cond = audio_cond + text_cond
        # cond = self.ca(audio_cond, text_cond)
        
        # film
        if text_cond is None:
            # text_cond = torch.randn(audio_cond.shape, device=x.device)  # 生成高斯噪声
            text_cond = torch.randn(audio_cond.shape[0], audio_cond.shape[1], 1, device=x.device)  # 生成高斯噪声
            # text_cond = torch.zeros(audio_cond.shape, device=x.device)  # 生成全0张量
        # 处理条件特征
        audio_cond = audio_cond.permute(0, 2, 1)  # [B, n_chans, T] -> [B, T, n_chans]
        text_cond = text_cond.permute(0, 2, 1)  # [B, n_chans, T] -> [B, T, n_chans]
        
        # 计算调制因子
        audio_modulation = self.audio_modulation(audio_cond).permute(0, 2, 1)  # [B, T, n_chans] -> [B, n_chans, T]
        text_modulation = self.text_modulation(text_cond).permute(0, 2, 1)  # [B, T, n_chans] -> [B, n_chans, T]

        # 融合 audio_cond 和 text_cond
        cond = audio_modulation + text_modulation  # [B, T, n_chans]
        
        # audio_cond = audio_cond.permute(0, 2, 1)  # [B, n_chans, T] -> [B, T, n_chans]
        # audio_modulation = self.audio_modulation(audio_cond).permute(0, 2, 1)  # [B, T, n_chans] -> [B, n_chans, T]
        # if text_cond is None:
        #     cond = audio_modulation
        # else:
        #     # 处理条件特征
        #     text_cond = text_cond.permute(0, 2, 1)  # [B, n_chans, T] -> [B, T, n_chans]
        #     text_modulation = self.text_modulation(text_cond).permute(0, 2, 1)  # [B, T, n_chans] -> [B, n_chans, T]
        #     cond = audio_modulation + text_modulation  # [B, T, n_chans]
        
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x[:, None, :, :]

        # if text_cond is None:
        #     # # 生成默认的text_cond，全0张量嵌入
        #     # text_cond = self.text_dafualt_embed(torch.zeros(1, dtype=torch.long, device=x.device))
        #     # text_cond = text_cond.repeat(audio_cond.shape[0], audio_cond.shape[-1], 1)
        #     # text_cond = text_cond.permute(0, 2, 1) # (1, 256, seq_len)
            
        #     # 生成高斯噪声，假设 text_cond 的维度为 (batch_size, seq_length, embed_dim)
        #     text_cond = torch.randn(audio_cond.shape, device=x.device)  # 生成高斯噪声
        # else:
        #     # text_cond = text_cond.repeat(1, 1, audio_cond.shape[-1])
        #     text_cond = text_cond.expand(-1, -1, audio_cond.shape[-1])
        # # cond = audio_cond + text_cond
        # cond = self.ca(audio_cond, text_cond)
        # if text_cond is None:
        #     text_cond = torch.randn(audio_cond.shape[0], audio_cond.shape[1], 1, device=x.device)  # 生成高斯噪声
        # gamma = self.gamma(text_cond)
        # beta = self.beta(text_cond)
        # # Reshape gamma and beta to match the dimensions of x: [B, 1, T] -> [B, C, T]
        # gamma = gamma.unsqueeze(2)
        # beta = beta.unsqueeze(2)
        # # Apply feature-wise linear modulation
        # cond = audio_cond * gamma + beta

class ControlWaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.text_dafualt_embed = nn.Embedding(1, n_hidden) 
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.ca = CrossAttention(n_hidden)
        self.residual_layers = nn.ModuleList([
            ResidualBlockFiLM(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=1
            )
            for i in range(n_layers)
        ])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)
        
        # 映射层用于调制因子
        self.audio_modulation = nn.Linear(n_hidden, n_hidden)
        self.singer_modulation = nn.Linear(n_hidden, n_hidden)
        
    def forward(self, spec, diffusion_step, audio_cond, singer_cond):
        """
        :param spec: [B, 1, D, T]
        :param diffusion_step: [B, 1]
        :param audio_cond: [B, D, T]
        :return:
        """
        B, D, T = audio_cond.shape
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        
        # adain
        if singer_cond is None:
            singer_cond = torch.randn(B, T, D, device=x.device)
        
        for layer in self.residual_layers:
            x, skip_connection = layer(x, audio_cond, singer_cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x[:, None, :, :]
    
class TextControlWaveNetNew(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.text_dafualt_embed = nn.Embedding(1, n_hidden) 
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.ca = CrossAttention(n_hidden)
        self.residual_layers = nn.ModuleList([
            ResidualBlockNew(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=1
            )
            for i in range(n_layers)
        ])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)
        
        # 映射层用于调制因子
        self.audio_modulation = nn.Linear(n_hidden, n_hidden)
        self.text_modulation = nn.Linear(n_hidden, n_hidden)
        
    def forward(self, spec, diffusion_step, audio_cond, text_cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param audio_cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        
        # film
        if text_cond is None:
            text_cond = torch.randn(audio_cond.shape[0], audio_cond.shape[1], device=x.device)  # 生成高斯噪声
            # text_cond = torch.zeros(audio_cond.shape[0], audio_cond.shape[1], 1, device=x.device)  # 生成全0张量
        
        for layer in self.residual_layers:
            x, skip_connection = layer(x, audio_cond, text_cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x[:, None, :, :]

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
        # 添加条件融合层
        self.cond_proj = nn.Linear(dim, dim)
        self.cond_norm = nn.LayerNorm(dim)

    def forward(self, x, audio_cond, diffusion_step):
        # 自注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # 条件融合
        cond = self.cond_proj(audio_cond)
        cond = self.cond_norm(cond)
        x = x + cond
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, in_dims=128, n_layers=12, n_chans=384, n_hidden=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.in_dims = in_dims
        self.n_chans = n_chans

        self.input_projection = nn.Linear(in_dims, n_chans)
        
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans)
        )
        
        self.audio_cond_projection = nn.Linear(n_hidden, n_chans)
        self.text_cond_projection = nn.Linear(n_hidden, n_chans)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(n_chans, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(n_chans, in_dims)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, spec, diffusion_step, audio_cond):        
        x = spec.squeeze(1).transpose(1, 2)  # [B, T, M]
        x = self.input_projection(x)  # [B, T, n_chans]
        
        time_emb = self.time_embedding(diffusion_step)  # [B, n_chans]
        x = x + time_emb.unsqueeze(1)  # 广播到所有时间步
        
        audio_cond = audio_cond.transpose(1, 2)  # [B, T, M]
        audio_cond = self.audio_cond_projection(audio_cond)  # [B, T, n_chans]
        
        # 在每个 Transformer 层中逐步融合条件
        for layer in self.transformer_layers:
            x = layer(x, audio_cond)
        
        x = self.output_projection(x)  # [B, T, M]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
        
        return x
    # def forward(self, spec, diffusion_step, audio_cond):        
    #     x = spec.squeeze(1).transpose(1, 2)  # [B, T, M]
    #     x = self.input_projection(x)  # [B, T, n_chans]
        
    #     time_emb = self.time_embedding(diffusion_step)  # [B, n_chans]
    #     x = x + time_emb.unsqueeze(1)  # 广播到所有时间步
        
    #     audio_cond = audio_cond.transpose(1, 2)  # [B, T, M]
    #     audio_cond = self.audio_cond_projection(audio_cond)  # [B, T, n_chans]
        
    #     # 在每个 Transformer 层中逐步融合条件
    #     for layer in self.transformer_layers:
    #         x = layer(x, audio_cond)
        
    #     x = self.output_projection(x)  # [B, T, M]
    #     x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
        
    #     return x
    

class TextControlTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
        # 添加条件融合层
        self.cond_proj = nn.Linear(dim * 2, dim)
        self.cond_norm = nn.LayerNorm(dim)

    def forward(self, x, audio_cond, text_cond):
        # 自注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # 条件融合
        cond = torch.cat([audio_cond, text_cond], dim=-1)
        cond = self.cond_proj(cond)
        cond = self.cond_norm(cond)
        x = x + cond
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TextControlDiffusionTransformer(nn.Module):
    def __init__(self, in_dims=128, n_layers=12, n_chans=384, n_hidden=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.in_dims = in_dims
        self.n_chans = n_chans

        self.input_projection = nn.Linear(in_dims, n_chans)
        
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans)
        )
        
        self.audio_cond_projection = nn.Linear(n_hidden, n_chans)
        self.text_cond_projection = nn.Linear(n_hidden, n_chans)
        
        self.transformer_layers = nn.ModuleList([
            TextControlTransformerBlock(n_chans, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(n_chans, in_dims)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, spec, diffusion_step, audio_cond, text_cond):
        B, _, M, T = spec.shape
        
        x = spec.squeeze(1).transpose(1, 2)  # [B, T, M]
        x = self.input_projection(x)  # [B, T, n_chans]
        
        time_emb = self.time_embedding(diffusion_step)  # [B, n_chans]
        x = x + time_emb.unsqueeze(1)  # 广播到所有时间步
        
        audio_cond = audio_cond.transpose(1, 2)  # [B, T, M]
        audio_cond = self.audio_cond_projection(audio_cond)  # [B, T, n_chans]
        
        if text_cond is not None:
            text_cond = text_cond.squeeze(-1)  # [B, M]
            text_cond = self.text_cond_projection(text_cond)  # [B, n_chans]
            text_cond = text_cond.unsqueeze(1).expand(-1, T, -1)  # [B, T, n_chans]
        else:
            text_cond = torch.zeros_like(audio_cond)
        
        # 在每个 Transformer 层中逐步融合条件
        for layer in self.transformer_layers:
            x = layer(x, audio_cond, text_cond)
        
        x = self.output_projection(x)  # [B, T, M]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
        
        return x

# class TransformerBlockNew(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout)
#         )
        
#         # 添加条件融合层
#         self.cond_proj = nn.Linear(dim, dim)
#         self.cond_norm = nn.LayerNorm(dim)

#     def forward(self, x, audio_cond):
#         # 自注意力
#         x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
#         cond = self.cond_proj(audio_cond)
#         cond = self.cond_norm(audio_cond)
#         x = x + cond
        
#         # MLP
#         x = x + self.mlp(self.norm2(x))
#         return x
    
# class DiffusionTransformerNew(nn.Module):
#     def __init__(self, in_dims=128, n_layers=12, n_chans=384, n_hidden=256, n_heads=8, dropout=0.1):
#         super().__init__()
#         self.in_dims = in_dims
#         self.n_chans = n_chans

#         self.input_projection = nn.Linear(in_dims, n_chans)
        
#         self.time_embedding = nn.Sequential(
#             SinusoidalPosEmb(n_chans),
#             nn.Linear(n_chans, n_chans * 4),
#             nn.GELU(),
#             nn.Linear(n_chans * 4, n_chans)
#         )
        
#         self.audio_cond_projection = nn.Linear(n_hidden, n_chans)
#         self.text_cond_projection = nn.Linear(n_hidden, n_chans)
        
#         self.transformer_layers = nn.ModuleList([
#             TransformerBlockNew(n_chans, n_heads, dropout=dropout)
#             for _ in range(n_layers)
#         ])
        
#         self.output_projection = nn.Linear(n_chans, in_dims)
#         nn.init.zeros_(self.output_projection.weight)
#         nn.init.zeros_(self.output_projection.bias)

#     def forward(self, spec, diffusion_step, cond):
#         B, _, M, T = spec.shape
        
#         x = spec.squeeze(1).transpose(1, 2)  # [B, T, M]
#         x = self.input_projection(x)  # [B, T, n_chans]
        
#         time_emb = self.time_embedding(diffusion_step)  # [B, n_chans]
#         x = x + time_emb.unsqueeze(1)  # 广播到所有时间步
        
#         cond = cond.transpose(1, 2)  # [B, T, M]
#         cond = self.audio_cond_projection(cond)  # [B, T, n_chans]
        
#         # if text_cond is not None:
#         #     text_cond = text_cond.squeeze(-1)  # [B, M]
#         #     text_cond = self.text_cond_projection(text_cond)  # [B, n_chans]
#         #     text_cond = text_cond.unsqueeze(1).expand(-1, T, -1)  # [B, T, n_chans]
#         # else:
#         #     text_cond = torch.zeros_like(audio_cond)
        
#         # 在每个 Transformer 层中逐步融合条件
#         for layer in self.transformer_layers:
#             x = layer(x, cond)
        
#         x = self.output_projection(x)  # [B, T, M]
#         x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
        
#         return x


class DecoupledCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k_audio = nn.Linear(dim, dim)
        self.v_audio = nn.Linear(dim, dim)
        self.k_singer = nn.Linear(dim, dim)
        self.v_singer = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, audio_context, singer_context=None, lambda_cond=1.0):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        
        # Audio cross-attention
        k_audio = self.k_audio(audio_context).reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v_audio = self.v_audio(audio_context).reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        
        attn_audio = (q @ k_audio.transpose(-2, -1)) * self.scale
        attn_audio = attn_audio.softmax(dim=-1)
        attn_audio = self.dropout(attn_audio)
        x_audio = (attn_audio @ v_audio).transpose(1, 2).reshape(b, n, c)
        
        # Text cross-attention (if text context is provided)
        if singer_context is not None:
            k_singer = self.k_singer(singer_context).reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
            v_singer = self.v_singer(singer_context).reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
            
            attn_singer = (q @ k_singer.transpose(-2, -1)) * self.scale
            attn_singer = attn_singer.softmax(dim=-1)
            attn_singer = self.dropout(attn_singer)
            x_singer = (attn_singer @ v_singer).transpose(1, 2).reshape(b, n, c)
            
            # Combine audio and condition attention
            x = x_audio + lambda_cond * x_singer
        else:
            x = x_audio
        
        return self.proj(x)

class TransformerBlockNew(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = DecoupledCrossAttention(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, audio_cond, singer_cond=None, lambda_singer=10):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.cross_attn(self.norm2(x), audio_cond, singer_cond, lambda_singer)
        x = x + self.mlp(self.norm3(x))
        return x

class DiffusionTransformerNew(nn.Module):
    '''Similar to IP-Adaptor, use decoupled cross-attention to fuse audio and text conditions'''
    def __init__(self, in_dims=128, n_layers=12, n_chans=384, n_hidden=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.in_dims = in_dims
        self.n_chans = n_chans

        self.input_projection = nn.Linear(in_dims, n_chans)
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.audio_cond_projection = nn.Linear(n_hidden, n_chans)
        self.singer_cond_projection = nn.Linear(n_hidden, n_chans)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlockNew(n_chans, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(n_chans, in_dims)
        self.norm = nn.LayerNorm(n_hidden)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, spec, diffusion_step, audio_cond, singer_cond=None, lambda_singer=1.0):
        B, _, M, T = spec.shape
        
        x = spec.squeeze(1).transpose(1, 2)  # [B, T, M]
        x = self.input_projection(x)  # [B, T, n_chans]
        
        time_emb = self.time_embedding(diffusion_step)  # [B, n_chans]
        x = x + time_emb.unsqueeze(1)  # 广播到所有时间步
        
        audio_cond = self.audio_cond_projection(audio_cond.transpose(1, 2))  # [B, T, n_chans]
        
        if singer_cond is not None:
            singer_cond = self.norm(singer_cond)
            singer_cond = self.singer_cond_projection(singer_cond)  # [B, n_chans]
        else:
            singer_cond = torch.randn_like(audio_cond)
        for layer in self.transformer_layers:
            x = layer(x, audio_cond, singer_cond, lambda_singer)
        
        x = self.output_projection(x)  # [B, T, M]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
        
        return x

    def init_singer_layers(self):
        for layer in self.transformer_layers:
            layer.cross_attn.k_singer.weight.data.copy_(layer.cross_attn.k_audio.weight.data)
            layer.cross_attn.k_singer.bias.data.copy_(layer.cross_attn.k_audio.bias.data)
            layer.cross_attn.v_singer.weight.data.copy_(layer.cross_attn.v_audio.weight.data)
            layer.cross_attn.v_singer.bias.data.copy_(layer.cross_attn.v_audio.bias.data)

    def freeze_audio_layers(self):
        for param in self.parameters():
            param.requires_grad = False
        
        # 解冻文本相关层
        for layer in self.transformer_layers:
            for param in layer.cross_attn.k_singer.parameters():
                param.requires_grad = True
            for param in layer.cross_attn.v_singer.parameters():
                param.requires_grad = True
        for param in self.singer_cond_projection.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
            
class AdaLNZero(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.scale1 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.scale2 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.shift1 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.shift2 = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x, cond):
        scale1, shift1, scale2, shift2 = cond.chunk(4, dim=-1)
        x = x + self.mha(self.norm1(x * (scale1 + 1) + shift1), self.norm1(x * (scale1 + 1) + shift1), self.norm1(x * (scale1 + 1) + shift1))[0]
        x = x + self.ff(self.norm2(x * (scale2 + 1) + shift2))
        return x

class DiTBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ada_ln_zero = AdaLNZero(d_model, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 4),  # 3倍d_model用于时间嵌入、音频条件和文本条件
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4)
        )

    def forward(self, x, time_emb, audio_cond, text_cond):
        cond = torch.cat([time_emb, audio_cond, text_cond], dim=-1)
        cond = self.mlp(cond)
        return self.ada_ln_zero(x, cond)

class LatentDiffusionTransformer(nn.Module):
    def __init__(self, in_dims=128, n_layers=12, n_heads=8, d_model=384):
        super().__init__()
        self.in_dims = in_dims
        self.d_model = d_model

        self.input_projection = nn.Linear(in_dims, d_model)
        self.time_embedding = SinusoidalPosEmb(d_model)
        self.audio_cond_projection = nn.Linear(in_dims, d_model)
        self.text_cond_projection = nn.Linear(in_dims, d_model)

        self.dit_blocks = nn.ModuleList([DiTBlock(d_model, n_heads) for _ in range(n_layers)])
        
        self.output_projection = nn.Linear(d_model, in_dims)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, spec, diffusion_step, audio_cond, text_cond):
        B, _, M, T = spec.shape
        
        # Linear and Reshape
        x = spec.view(B, -1, self.in_dims)
        x = self.input_projection(x)
        
        # Embeddings and conditions
        time_emb = self.time_embedding(diffusion_step.squeeze(1))
        audio_cond = self.audio_cond_projection(audio_cond.transpose(1, 2))  # [B, T, d_model]
        
        if text_cond is not None:
            text_cond = self.text_cond_projection(text_cond.squeeze(-1))  # [B, d_model]
            text_cond = text_cond.unsqueeze(1).expand(-1, T, -1)  # [B, T, d_model]
        else:
            text_cond = torch.zeros_like(audio_cond)
        
        # DiT Blocks
        for block in self.dit_blocks:
            x = block(x, time_emb.unsqueeze(1).expand(-1, T, -1), audio_cond, text_cond)
        
        # Output projection
        x = self.output_projection(x)
        
        # Reshape back to original dimensions
        x = x.view(B, 1, M, T)
        
        return x

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class TextControlWaveUNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=12, channels_interval=24):
        super().__init__()
        
        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.diffusion_embedding = SinusoidalPosEmb(self.n_layers * self.channels_interval)  # Ensure it's defined
        self.diffusion_proj = nn.Sequential(
            nn.Linear(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval * 4),
            Mish(),  # Ensure it's defined
            nn.Linear(self.n_layers * self.channels_interval * 4, self.n_layers * self.channels_interval)
        )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1, padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input, diffusion_step, audio_cond, text_cond=None):
        """
        :param input: [B, 1, M, T] - Input tensor
        :param diffusion_step: [B, 1] - Diffusion step
        :param audio_cond: [B, M, T] - Audio conditioning
        :param text_cond: [B, M, T] - Text conditioning (optional)
        """
        x = input.squeeze(1)  # Now [B, M, T]
        
        # Encoder
        tmp = []
        for i in range(self.n_layers):
            x = self.encoder[i](x)
            tmp.append(x)
            x = x[:, :, ::2]
        
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_proj = self.diffusion_proj(diffusion_emb)
        x = torch.cat([x, diffusion_proj], dim=1)
        x = self.middle(x)
        
        # Decoder
        for i in range(self.n_layers):
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=True)
            
            # FiLM fusion with audio conditioning
            if audio_cond is not None:
                audio_scale = audio_cond[:, :x.size(1), :x.size(2)]  # Adjust to match dimensions
                audio_bias = audio_cond[:, x.size(1):2*x.size(1), :x.size(2)]
                x = x * audio_scale + audio_bias
            
            # Handle text conditioning
            if text_cond is not None:
                text_scale = text_cond[:, :x.size(1), :x.size(2)]
                text_bias = text_cond[:, x.size(1):2*x.size(1), :x.size(2)]
                x = x * text_scale + text_bias
            else:
                # Gaussian random sampling for text_cond
                batch_size, channels, length = x.size()
                text_scale = torch.randn(batch_size, channels, length, device=x.device)
                text_bias = torch.randn(batch_size, channels, length, device=x.device)
                x = x * text_scale + text_bias
            
            x = torch.cat([x, tmp[self.n_layers - i - 1]], dim=1)
            x = self.decoder[i](x)
        
        x = torch.cat([x, input], dim=1)  # Ensure dimensions match
        x = self.out(x)
        
        return x[:, None, :, :]  # Confirm output dimensions

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, temperature=1):
        super(CrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)
        self.temperature = temperature
        
    def forward(self, feature1, feature2):
        # feature1作为Query
        query = self.query_layer(feature1.permute(0, 2, 1))
        # feature2作为Key和Value
        key = self.key_layer(feature2.permute(0, 2, 1))
        value = self.value_layer(feature2.permute(0, 2, 1))

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / self.temperature
        attention_scores = F.softmax(attention_scores, dim=-1)
        # 应用注意力权重到Value上
        weighted_values = torch.matmul(attention_scores, value)

        return weighted_values.permute(0, 2, 1)