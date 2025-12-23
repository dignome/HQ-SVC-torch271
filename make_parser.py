# make_parse.py

import argparse


def make_parse_train():
    parser = argparse.ArgumentParser(description='Singing Classifier Training')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample_rate', default='44100', type=int)
    parser.add_argument('--n_mels', default=128, type=int)
    parser.add_argument('--hop_length', default=512, type=int)
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--ckpt_dir', default=None, type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--k_fold', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids to use')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--epochs_per_save', default=1, type=int)
    parser.add_argument('--steps_per_print', type=int, default=1000)
    parser.add_argument('--num_log_audio', type=int, default=2)
    parser.add_argument('--log_epoch', type=int, default=1, help='Log interval')
    parser.add_argument('--log_steps', type=int, default=100, help='Log interval')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--loss_fn', choices=['ce', 'fl', 'dl'], default='ce', help='Loss function')
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--optimizer',
                        choices=['batch_gd', 'online_gd', 'mini_batch_gd', 'sgd', 'momentum', 'adagrad', 'adam','admax'], default='sgd', help='Optimizer')
    parser.add_argument('--hifigan_config_path', type=str, default='configs/hifigan_config.json', help='HiFi-GAN config')
    return parser

def make_parse_test():
    parser = argparse.ArgumentParser(description='MNIST Classifier Testing')
    parser.add_argument('--model', help='Model')
    parser.add_argument('--data_path', type=str, default='../datasets/mnist', help='Data path')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--ckpt_dir', type=str, default='simple_classifier.pth', help='Checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    return parser

def make_parse_hifigan():
    parser = argparse.ArgumentParser(description='HiFi-GAN training config')
    
    # 添加参数，这些参数可以根据实际需要进行修改或添加
    parser.add_argument('--segment_size', type=int, default=4, 
                        help='The length of audio to process at one time in seconds.')
    parser.add_argument('--n_fft', type=int, default=1024, 
                        help='The number of points to use in each block for the FFT.')
    parser.add_argument('--num_mels', type=int, default=128, 
                        help='The number of Mel filters to use for the Mel-spectrogram.')
    parser.add_argument('--hop_size', type=int, default=256, 
                        help='The number of samples between successive frames.')
    parser.add_argument('--win_size', type=int, default=1024, 
                        help='The size of the window to use for each frame.')
    parser.add_argument('--sampling_rate', type=int, default=22050, 
                        help='The sampling rate of the audio.')
    parser.add_argument('--fmin', type=int, default=40, 
                        help='The minimum frequency to include in the Mel-spectrogram.')
    parser.add_argument('--fmax', type=int, default=16000, 
                        help='The maximum frequency to include in the Mel-spectrogram.')
    parser.add_argument('--fmax_for_loss', type=int, default=16000, 
                        help='The maximum frequency to include in the loss Mel-spectrogram.')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='The random seed for PyTorch.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='The learning rate for the Adam optimizer.')
    parser.add_argument('--lr_decay', type=float, default=0.99997, 
                        help='The learning rate decay factor for the exponential learning rate scheduler.')
    parser.add_argument('--adam_b1', type=float, default=0.8, 
                        help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_b2', type=float, default=0.99, 
                        help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='The number of workers used in the data loader.')
    
    # 可以根据需要继续添加更多参数
    
    return parser

def make_parse_cmd():
    parser = argparse.ArgumentParser(description='Cmd for load config')
    parser.add_argument('--config',
                        '-c', 
                        type=str, 
                        required=True, 
                        default='configs/finetune_facodec_v1.3.yaml',
                        help='path to the config file')
    return parser
