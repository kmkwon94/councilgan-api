"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
from trainer_council import Council_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import torch
import os
import shutil
import uuid
from tqdm import tqdm
import time

def runImageTransfer(preload_model, input_folder, user_key, a2b):
    #preload_model = [trainer, config, council_size, style_dim]
    trainer = preload_model[0]
    config = preload_model[1]
    council_size = preload_model[2]
    style_dim = preload_model[3]

    output_path = 'static'
    seed = 1
    num_of_images_to_test = 100
    
    # Setup model and data loader
    image_names = ImageFolder(input_folder, transform=None, return_paths=True)
    if not 'new_size_a' in config.keys():
        config['new_size_a'] = config['new_size']
    is_data_A = a2b
    data_loader = get_data_loader_folder(input_folder, 1, False,\
                                        new_size=config['new_size_a'] if 'new_size_a' in config.keys() else config['new_size'],\
                                        crop=False, config=config, is_data_A=is_data_A)
                                        
    encode_s = []
    decode_s = []
    if a2b:
        for i in range(council_size):
            encode_s.append(trainer.gen_a2b_s[i].encode)  # encode function
            decode_s.append(trainer.gen_a2b_s[i].decode)  # decode function
    else:
        for i in range(council_size):
            encode_s.append(trainer.gen_b2a_s[i].encode)  # encode function
            decode_s.append(trainer.gen_b2a_s[i].decode)  # decode function
    
    # creat testing images
    file_list= [] 
    seed = 1
    curr_image_num = -1
    
    for i, (images, names) in tqdm(enumerate(zip(data_loader, image_names)), total=num_of_images_to_test):

        if curr_image_num == num_of_images_to_test:
            break
        curr_image_num += 1
        k = np.random.randint(council_size)
        style_fixed = Variable(torch.randn(10, style_dim, 1, 1).cuda(), volatile=True)
        print(names[1])
        images = Variable(images.cuda(), volatile=True)

        content, _ = encode_s[k](images)
        seed += 1
        torch.random.manual_seed(seed)
        style = Variable(torch.randn(10, style_dim, 1, 1).cuda(), volatile=True)
        
        for j in range(10):
            s = style[j].unsqueeze(0)
            outputs = decode_s[k](content, s, images)
            basename = os.path.basename(names[1])
            output_folder = os.path.join(output_path, 'img') #output_folder = static/img
                
            path_all_in_one = os.path.join(output_folder, user_key , '_out_' + str(curr_image_num) + '_' + str(j) + '.jpg')
            file_list.append(path_all_in_one)
            do_all_in_one = True
            if do_all_in_one:
                if not os.path.exists(os.path.dirname(path_all_in_one)):
                    os.makedirs(os.path.dirname(path_all_in_one))
            vutils.save_image(outputs.data, path_all_in_one, padding=0, normalize=True)
    return file_list