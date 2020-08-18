# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, after_this_request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from test_on_folder import runImageTransfer
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
from trainer_council import Council_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
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
import random
import string
import uuid
import shutil
from queue import Queue, Empty
from threading import Thread
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import base64
import io
import signal
import sys

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)
path = "./static"
threads = []

#multi-threads with return value
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, timeout=3):
        Thread.join(self, timeout=3)
        return self._return

class thread_with_trace(ThreadWithReturnValue):
    def __init__(self, *args, **keywords):
        ThreadWithReturnValue.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        ThreadWithReturnValue.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True
############################################################
#preload model
def loadModel(config, checkpoint, a2b):
    seed = 1

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Load experiment setting
    config = get_config(config)
    input_dim = config['input_dim_a'] if a2b else config['input_dim_b']
    council_size = config['council']['council_size']

    style_dim = config['gen']['style_dim']
    trainer = Council_Trainer(config)
    only_one = False
    if 'gen_' in checkpoint[-21:]:
        state_dict = torch.load(checkpoint)
        try:
            print(state_dict)
            if a2b:
                trainer.gen_a2b_s[0].load_state_dict(state_dict['a2b'])
            else:
                trainer.gen_b2a_s[0].load_state_dict(state_dict['b2a'])
        except:
            print('a2b should be set to ' + str(not a2b) + ' , Or config file could be wrong')
            a2b = not a2b
            if a2b:
                trainer.gen_a2b_s[0].load_state_dict(state_dict['a2b'])
            else:
                trainer.gen_b2a_s[0].load_state_dict(state_dict['b2a'])
                
        council_size = 1
        only_one = True
    else:
        for i in range(council_size):
            try:
                if a2b:
                    tmp_checkpoint = checkpoint[:-8] + 'a2b_gen_' + str(i) + '_' + checkpoint[-8:] + '.pt'
                    state_dict = torch.load(tmp_checkpoint)
                    trainer.gen_a2b_s[i].load_state_dict(state_dict['a2b'])
                else:
                    tmp_checkpoint = checkpoint[:-8] + 'b2a_gen_' + str(i) + '_' + checkpoint[-8:] + '.pt'
                    state_dict = torch.load(tmp_checkpoint)
                    trainer.gen_b2a_s[i].load_state_dict(state_dict['b2a'])
            except:
                print('a2b should be set to ' + str(not a2b) + ' , Or config file could be wrong')
                
                a2b = not a2b
                if a2b:
                    tmp_checkpoint = checkpoint[:-8] + 'a2b_gen_' + str(i) + '_' + checkpoint[-8:] + '.pt'
                    state_dict = torch.load(tmp_checkpoint)
                    trainer.gen_a2b_s[i].load_state_dict(state_dict['a2b'])
                else:
                    tmp_checkpoint = checkpoint[:-8] + 'b2a_gen_' + str(i) + '_' + checkpoint[-8:] + '.pt'
                    state_dict = torch.load(tmp_checkpoint)
                    trainer.gen_b2a_s[i].load_state_dict(state_dict['b2a'])

    trainer.cuda()
    trainer.eval()

    return [trainer, config, council_size, style_dim]

peson2anime_preloadModel = loadModel("pretrain/anime/256/anime2face_council_folder.yaml", "pretrain/anime/256/01000000", 0)
male2female_preloadModel = loadModel("pretrain/m2f/256/male2female_council_folder.yaml", "pretrain/m2f/256/01000000", 1)
noglasses_preloadModel = loadModel("pretrain/glasses_removal/128/glasses_council_folder.yaml", "pretrain/glasses_removal/128/01000000", 1)
############################################################

# 업로드 HTML 렌더링
@app.route('/')
def render_file():
    return render_template('upload.html')

@app.route('/healthz', methods=['GET'])
def healthz():
    return "I am alive", 200

@app.route('/fileUpload', methods=['POST'])
def fileupload():
    #내가 전달 받는 request는 'file'과 'check_model'
    check_value = request.form['check_model']
    f = request.files['file']
    
    #handling over request by status code 429
    global threads
    print(len(threads),"fileUpload")
    if len(threads) > 10:
        return Response("error : Too many requests", status=429)
    
    try:
        randomDirName = str(uuid.uuid4()) #사용자끼리의 업로드한 이미지가 겹치지 않게끔 uuid를 이용하여 사용자를 구분하는 디렉터리를 만든다.
        if check_value == "ani":
            path = '/home/user/upload/person2anime/'
            os.mkdir(path + randomDirName)
            f.save(path + randomDirName + '/' + secure_filename(f.filename))
            byte_image_list = person_To_anime(randomDirName)
            return render_template('showImage.html', rawimg=byte_image_list)
        elif check_value == "m2f":
            path = '/home/user/upload/male2female/'
            os.mkdir(path + randomDirName)
            f.save(path + randomDirName + '/' +secure_filename(f.filename))
            byte_image_list = male_To_female(randomDirName)
            return render_template('showImage.html', rawimg=byte_image_list)
        else:
            path = '/home/user/upload/no_glasses/'
            os.mkdir(path + randomDirName)
            f.save(path + randomDirName + '/' + secure_filename(f.filename))
            byte_image_list = no_glasses(randomDirName)
            return render_template('showImage.html', rawimg=byte_image_list)
    except Exception as e:
        print(e)
        return Response("upload file and load model is fail", status=400)

@app.route('/convert_image', methods=['POST'])
def convert_imgae():
    check_value = request.form['check_model']
    f = request.files['file']
    
    #handling over request by status code 429
    global threads
    print(len(threads),"convert_image")
    if len(threads) > 10:
        return Response("error : Too many requests", status=429)
    
    try:
        randomDirName = str(uuid.uuid4()) #사용자끼리의 업로드한 이미지가 겹치지 않게끔 uuid를 이용하여 사용자를 구분하는 디렉터리를 만든다.
        if check_value == "ani":
            path = '/home/user/upload/person2anime/'
            os.mkdir(path + randomDirName)
            f.save(path + randomDirName + '/' + secure_filename(f.filename))
            byte_image_list = person_To_anime(randomDirName)
            return send_file(byte_image_list, mimetype="image/jpeg")
        elif check_value == "m2f":
            path = '/home/user/upload/male2female/'
            os.mkdir(path + randomDirName)
            f.save(path + randomDirName + '/' +secure_filename(f.filename))
            byte_image_list = male_To_female(randomDirName)
            return send_file(byte_image_list, mimetype="image/jpeg")
        else:
            path = '/home/user/upload/no_glasses/'
            os.mkdir(path + randomDirName)
            f.save(path + randomDirName + '/' + secure_filename(f.filename))
            byte_image_list = no_glasses(randomDirName)
            return send_file(byte_image_list, mimetype="image/jpeg")
    except Exception as e:
        print(e)
        return Response("upload file and load model is fail", status=400)

#사용자의 입력을 받아서 각 원하는 결과물을 라우팅
def person_To_anime(randomDirName):
    try:
        user_key = randomDirName
        input_ = "/home/user/upload/person2anime/" + user_key
        a2b = 0
        model_type = 'person2anime'
      
        #handling multi-threads
        t1 = thread_with_trace(target=runImageTransfer, args=(peson2anime_preloadModel,input_,user_key,a2b))
        t1.user_id = user_key
        threads.append(t1)
        while threads[0].user_id!=user_key:
            print(str(user_key)+": ", threads[0].user_id)
            if threads[0].is_alive():
                threads[0].join()
        threads[0].start()
        
        file_list = threads[0].join(timeout=3)
        if threads[0].is_alive():
            threads[0].kill()
            threads.pop(0)
            raise Exception("error model does not work! please try again 30 seconds later")
        threads.pop(0)
        file_list.sort()
        
        byte_image_list = [] #byte_image를 담기위한 list
        tmp_list = [] #byte_image를 담기전에 decode 하기 위한 list
        #imgFIle은 np.array형태여야 fromarray에 담길수 있음
        #img_io는 각 파일마다 byte 객체를 적용해줘야하므로 for문 안에서 같이 반복을 돌아야 함
        #b64encode로 encode 해준다.
        for image in file_list:
            imgFile = PIL.Image.fromarray(np.array(PIL.Image.open(image).convert("RGB")))
            img_io = io.BytesIO()
            imgFile.save(img_io, 'jpeg', quality = 100)
            img_io.seek(0)
            img = base64.b64encode(img_io.getvalue())
            tmp_list.append(img)
        
        #decode 작업은 여기서 해준다.
        for i in tmp_list:
            byte_image_list.append(i.decode('ascii'))
        
        #input file과 output file을 모두 제거해주는 함수 호출
        remove(user_key, model_type)
        return byte_image_list
        #return render_template('showImage.html', rawimg=byte_image_list)
    except Exception as e:
        print(e)
        return Response("person2anime is fail", status=400)    

def male_To_female(randomDirName):
    try:
        user_key = randomDirName
        input_ = "/home/user/upload/male2female/" + user_key
        a2b = 1
        model_type = 'male2female'

        t1 = thread_with_trace(target=runImageTransfer, args=(male2female_preloadModel,input_,user_key,a2b))
        t1.user_id = user_key
        threads.append(t1)
        while threads[0].user_id!=user_key:
            print(str(user_key)+": ", threads[0].user_id)
            if threads[0].is_alive():
                threads[0].join()
        threads[0].start()

        file_list = threads[0].join(timeout=3)
        if threads[0].is_alive():
            threads[0].kill()
            threads.pop(0)
            raise Exception("error model does not work! please try again 30 seconds later")
        threads.pop(0)
        file_list.sort()
        
        byte_image_list = [] #byte_image를 담기위한 list
        tmp_list = [] #byte_image를 담기전에 decode 하기 위한 list

        #imgFIle은 np.array형태여야 fromarray에 담길수 있음
        #img_io는 각 파일마다 byte 객체를 적용해줘야하므로 for문 안에서 같이 반복을 돌아야 함
        #b64encode로 encode 해준다.
        for image in file_list:
            imgFile = PIL.Image.fromarray(np.array(PIL.Image.open(image).convert("RGB")))
            img_io = io.BytesIO()
            imgFile.save(img_io, 'jpeg', quality = 100)
            img_io.seek(0)
            img = base64.b64encode(img_io.getvalue())
            tmp_list.append(img)
        
        #decode 작업은 여기서 해준다.
        for i in tmp_list:
            byte_image_list.append(i.decode('ascii'))
        
        #input file과 output file을 모두 제거해주는 함수 호출
        remove(user_key, model_type)
        return byte_image_list
        #return render_template('showImage.html', rawimg=byte_image_list)
    except Exception as e:
        print(e)
        return Response("male2female is fail", status=400)    
   
def no_glasses(randomDirName):
    try:
        user_key = randomDirName
        input_ = "/home/user/upload/no_glasses/" + user_key
        a2b = 1
        model_type = 'no_glasses'

        t1 = thread_with_trace(target=runImageTransfer, args=(noglasses_preloadModel,input_,user_key,a2b))
        t1.user_id = user_key
        threads.append(t1)
        while threads[0].user_id!=user_key:
            print(str(user_key)+": ", threads[0].user_id)
            if threads[0].is_alive():
                threads[0].join()
        threads[0].start()
        
        file_list = threads[0].join(timeout=3)   
        if threads[0].is_alive():
            threads[0].kill()
            threads.pop(0)
            raise Exception("error model does not work! please try again 30 seconds later")
        threads.pop(0)
        file_list.sort()
        
        byte_image_list = [] #byte_image를 담기위한 list
        tmp_list = [] #byte_image를 담기전에 decode 하기 위한 list
        
        #imgFIle은 np.array형태여야 fromarray에 담길수 있음
        #img_io는 각 파일마다 byte 객체를 적용해줘야하므로 for문 안에서 같이 반복을 돌아야 함
        #b64encode로 encode 해준다.
        for image in file_list:
            imgFile = PIL.Image.fromarray(np.array(PIL.Image.open(image).convert("RGB")))
            img_io = io.BytesIO()
            imgFile.save(img_io, 'jpeg', quality = 100)
            img_io.seek(0)
            img = base64.b64encode(img_io.getvalue())
            tmp_list.append(img)
        
        #decode 작업은 여기서 해준다.
        for i in tmp_list:
            byte_image_list.append(i.decode('ascii'))
        
        #input file과 output file을 모두 제거해주는 함수 호출
        remove(user_key, model_type)
        return byte_image_list
        #return render_template('showImage.html', rawimg=byte_image_list)
    except Exception as e:
        print(e)
        return Response("no_glasses is fail", status=400)    

def remove(user_key, model_type):
    remove_input_dir = '/home/user/upload/' + model_type + '/' + user_key 
    path = os.path.join('static/img/', user_key)
    print("Now start to remove file")
    print("user key is " + user_key)
    print("Input path " + remove_input_dir)
    print("Output path " + path)

    #output path를 삭제하는 try 문
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print("Delete " + path + " is completed")
    except Exception as e:
        print(e)
        print("Delete" + path + " is failed")
    #input path를 삭제하는 try 문
    try:
        if os.path.isdir(remove_input_dir):
            shutil.rmtree(remove_input_dir)
            print("Delete" + remove_input_dir + " is completed")
    except Exception as e:
        print("Delete" + remove_input_dir + " is failed")
    
    return print("All of delete process is completed!")

if __name__ == '__main__':
    # server execute
    app.run(host='0.0.0.0', port=80)