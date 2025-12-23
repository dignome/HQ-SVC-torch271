'''
author: wayn391@mastertones
'''

import os
import json
import time
import yaml
import datetime
import torch
import matplotlib.pyplot as plt
from . import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
class Saver(object):
    def __init__(
            self, 
            args,
            initial_global_step=0):
        
        # cold start
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()
        self.log_dir = args.log_dir
        self.sample_rate = args.sample_rate

        # ckpt
        os.makedirs(self.log_dir, exist_ok=True)       

        # writer
        self.writer = SummaryWriter(self.log_dir)


    def log_info(self, msg):
        '''log method'''
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                tmp_str = ''
                if isinstance(v, int):
                    tmp_str = '{}: {:,}'.format(k, v)
                else:
                    tmp_str = '{}: {}'.format(k, v)

                msg_list.append(tmp_str)
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg
        
        # dsplay
        print(msg_str)

        # save
        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str+'\n')

    def log_value(self, dict):
        for k, v in dict.items():
            self.writer.add_scalar(k, v, self.global_step)

    def log_spec(self, name, spec, vmin=-14, vmax=3.5):
    # 检查 spec 是否为 Tensor，并转换为 numpy
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()

        # 为 spec 绘制图像
        fig = plt.figure(figsize=(12, 6))
        # font_path = 'SimHei'  # 或者字体的绝对路径
        # font_prop = FontProperties(fname=font_path, size=14)
        plt.imshow(spec, aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar()
        # plt.title(name, fontproperties=font_prop)
        plt.gca().invert_yaxis()  # 反转y轴
        plt.tight_layout()
        
        # 将图像添加到 TensorBoard
        self.writer.add_figure(name, fig, self.global_step)
        
        # 关闭图形以释放资源
        plt.close(fig)
    
    def log_audio(self, dict):
        for k, v in dict.items():
            self.writer.add_audio(k, v, global_step=self.global_step, sample_rate=self.sample_rate)
    
    def get_interval_time(self, update=True):
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        if update:
            self.last_time = cur_time
        return time_interval

    def get_total_time(self, to_str=True):
        total_time = time.time() - self.init_time
        if to_str:
            total_time = str(datetime.timedelta(
                seconds=total_time))[:-5]
        return total_time

    def save_model(
            self,
            model, 
            optimizer,
            name='model',
            postfix='',
            to_json=False):
        # os.makedirs(os.path.join(self.expdir), exist_ok=True)
        # path
        if postfix:
            postfix = '_' + postfix
        path_pt = os.path.join(
            self.log_dir , name+postfix+'.pt')
       
        # check
        print(' [*] model checkpoint saved: {}'.format(path_pt))

        # save
        if optimizer is not None:
            torch.save({
                'global_step': self.global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path_pt)
        else:
            torch.save({
                'global_step': self.global_step,
                'model': model.state_dict()}, path_pt)
            
        # to json
        # if to_json:
        #     path_json = os.path.join(
        #         self.expdir , name+'.json')
        #     utils.to_json(path_params, path_json)
    
    def delete_model(self, name='model', postfix=''):
        # path
        if postfix:
            postfix = '_' + postfix
        path_pt = os.path.join(
            self.expdir , name+postfix+'.pt')
       
        # delete
        if os.path.exists(path_pt):
            os.remove(path_pt)
            print(' [*] model checkpoint deleted: {}'.format(path_pt))
        
    def global_step_increment(self):
        self.global_step += 1


