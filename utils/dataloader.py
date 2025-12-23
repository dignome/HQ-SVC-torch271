import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchaudio
from sklearn.model_selection import KFold
import glob
import numpy as np
from utils.utils import repeat_expand
# def get_file_name(path):
#     normalized_path = os.path.normpath(path)
#     path_parts = normalized_path.split(os.sep)
#     spk_name = path_parts[-2]
#     wav_name = path_parts[-1]
#     file_name = f'{spk_name}_{wav_name}'
#     return file_name

# # Define the dataset
# class AudioDataset(Dataset):
#     def __init__(self, audio_paths, transform=None):
#         """
#         初始化数据集。
#         :param audio_paths: 音频文件的路径列表。
#         :param transform: 应用于每个音频样本的可选变换。
#         """
#         self.audio_paths = audio_paths
#         self.transform = transform

#     def __len__(self):
#         """
#         返回数据集中样本的数量。
#         """
#         return len(self.audio_paths)

#     def __getitem__(self, idx):
#         """
#         根据索引获取音频样本。
#         """
#         audio_16k_path = self.audio_paths[idx]
#         mel_44k_path = audio_16k_path.replace('audio_16k', 'mel_44k').replace('.wav', '.npy')
        
#         # 加载音频文件
#         # wav, sr = torchaudio.load(audio_path)
#         wav_16k ,_ = torchaudio.load(audio_16k_path)
#         mel_44k = torch.from_numpy(np.load(mel_44k_path))
        
#         file_name = get_file_name(audio_16k_path)
        
#         return wav_16k, mel_44k, file_name

def get_file_name(path):
    normalized_path = os.path.normpath(path)
    path_parts = normalized_path.split(os.sep)
    try:
        # 尝试获取倒数第二个和最后一个部分作为说话者名和文件名
        spk_name = path_parts[-2]
        wav_name = path_parts[-1].split('.')[0]  # 移除文件扩展名
        file_name = f'{spk_name}_{wav_name}'
    except IndexError:
        # 如果路径格式不正确，返回原始路径
        file_name = path
    return file_name

def wav_pad(wav, multiple=200):
    batch, seq_len = wav.shape
    padded_len = ((seq_len + (multiple-1)) // multiple) * multiple
    padded_wav = repeat_expand(wav, padded_len)
    return padded_wav

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, transform=None):
        """
        初始化数据集。
        :param audio_paths: 音频文件的路径列表。
        :param transform: 应用于每个音频样本的可选变换。
        """
        self.audio_paths = audio_paths
        self.transform = transform

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        根据索引获取音频样本。
        """
        if idx >= len(self.audio_paths):
            raise IndexError("Index out of bounds")
        
        audio_16k_path = self.audio_paths[idx]
        mel_44k_path = os.path.splitext(audio_16k_path)[0].replace('audio_16k', 'mel_44k') + '.npy'
        audio_44k_path = os.path.splitext(audio_16k_path)[0].replace('audio_16k', 'audio_44k') + '.wav'
        vq_post_path = os.path.splitext(audio_16k_path)[0].replace('audio_16k', 'vq_post') + '.npy'
        spk_path = os.path.splitext(audio_16k_path)[0].replace('audio_16k', 'spk') + '.npy'
        wav_44k, _ = torchaudio.load(audio_44k_path)
        
        # For padding to muliple of 200
        wav_44k = wav_pad(wav_44k)

        # 确保mel_44k文件存在
        if not os.path.isfile(mel_44k_path):
            raise FileNotFoundError(f"Mel spectrogram file not found: {mel_44k_path}")
        
        mel_44k = torch.from_numpy(np.load(mel_44k_path))
        vq_post = torch.from_numpy(np.load(vq_post_path))
        spk = torch.from_numpy(np.load(spk_path))
        file_name = get_file_name(audio_16k_path)
        
        return wav_44k, mel_44k, vq_post, spk, file_name
    
def load_audio_data(data_path, batch_size=64, validation_ratio=0.1, demo_num=0):
    
    # train_audio_paths = [os.path.join(data_path, 'train', f) for f in os.listdir(data_path) if f.endswith('.wav')]
    # test_audio_paths = [os.path.join(data_path, 'test', f) for f in os.listdir(data_path) if f.endswith('.wav')]
    
    train_audio_16k_paths = glob.glob(os.path.join(data_path, 'train', 'audio_16k/**', '*.wav'))
    test_audio_16k_paths = glob.glob(os.path.join(data_path, 'test', 'audio_16k/**', '*.wav'))
    
    if type(demo_num) is int and demo_num > 0:
        try:
            train_audio_16k_paths = train_audio_16k_paths[:demo_num]
            test_audio_16k_paths = test_audio_16k_paths[:demo_num//9]
        except:
            raise ValueError
    dataset = AudioDataset(train_audio_16k_paths)
    
     # Split dataset into training and validation sets
    total_size = len(dataset)
    val_size = int(validation_ratio * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = AudioDataset(test_audio_16k_paths)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def load_audio_data_k_fold(data_path, k_folds=5, batch_size=64, demo_num=0):
    audio_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
    
    if type(demo_num) is int and demo_num > 0:
        try:
            audio_paths = audio_paths[:demo_num]
        except:
            raise ValueError
        
    # 创建一个完整的数据集
    full_dataset = AudioDataset(audio_paths)
    
    # 初始化KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_loaders = []

    for train_index, val_index in kf.split(full_dataset):
        # 根据索引分割数据集
        train_dataset = Subset(full_dataset, train_index)
        val_dataset = Subset(full_dataset, val_index)

        # 创建训练和验证的数据加载器
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        # 添加当前折的数据加载器到列表
        fold_loaders.append((train_loader, val_loader))

    # 测试集数据加载器不需要分割，使用全部数据
    test_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=False)

    return fold_loaders, test_loader

class FocalLoss:
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets)

        elif self.alpha_t is not None and self.gamma == 0: 
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha_t)

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha_t is not None and self.gamma != 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha_t, reduction='none')
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss