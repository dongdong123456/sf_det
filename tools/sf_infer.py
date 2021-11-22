# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import glob
from functools import reduce
import datetime
import time
import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from deploy.python.benchmark_utils import PaddleInferBenchmark
from deploy.python.picodet_postprocess import PicoDetPostProcess
from deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize
from deploy.python.visualize import visualize_box_mask
from deploy.python.utils import argsparser, Timer, get_current_memory_mb

import socket
from socket import *
from threading import Thread
import queue
from pathlib import Path
import numpy

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import random
import json

# 新建一个用户组
authorizer = DummyAuthorizer()
# 将用户名，密码，指定目录，权限 添加到里面
if not os.path.exists("D:\\FTP Root\\ICR_EXT"):
    os.mkdir("D:\\FTP Root\\ICR_EXT")
authorizer.add_user("sick", "sick", "D:\\FTP Root\\ICR_EXT", perm="elradfmw")  # adfmw
handler = FTPHandler
handler.authorizer = authorizer
handler.passive_ports = range(5001, 5510)

controler_tcpCliSock = socket(AF_INET, SOCK_STREAM)

HOST_controler = '127.0.0.1'  # The remote host
PORT_controler = 2010  # The same port as used by the server

imagelist = queue.Queue()



# Global dictionary
SUPPORT_MODELS = {
    'YOLO',
    'RCNN',
    'SSD',
    'Face',
    'FCOS',
    'SOLOv2',
    'TTFNet',
    'S2ANet',
    'JDE',
    'FairMOT',
    'DeepSORT',
    'GFL',
    'PicoDet',
}


class Detector(object):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='fluid',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        return inputs

    def postprocess(self,
                    np_boxes,
                    np_masks,
                    inputs,
                    np_boxes_num,
                    threshold=0.86):
        # postprocess output of predictor
        results = {}
        results['boxes'] = np_boxes
        results['boxes_num'] = np_boxes_num
        if np_masks is not None:
            results['masks'] = np_masks
        return results

    def predict(self, image_list, threshold=0.86, warmup=0, repeats=1):
        '''
        Args:
            image_list (list): list of image
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_list)
        self.det_times.preprocess_time_s.end()
        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            boxes_num = self.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = {'boxes': np.array([[]]), 'boxes_num': [0]}
        else:
            results = self.postprocess(
                np_boxes, np_masks, inputs, np_boxes_num, threshold=threshold)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += len(image_list)
        return results

    def get_timer(self):
        return self.det_times


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir,
                   run_mode='fluid',
                   batch_size=1,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    if device != 'GPU' and run_mode != 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
            .format(run_mode, device))
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        config.enable_xpu(10 * 1024 * 1024)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)

        if use_dynamic_shape:
            min_input_shape = {
                'image': [batch_size, 3, trt_min_shape, trt_min_shape]
            }
            max_input_shape = {
                'image': [batch_size, 3, trt_max_shape, trt_max_shape]
            }
            opt_input_shape = {
                'image': [batch_size, 3, trt_opt_shape, trt_opt_shape]
            }
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images


def visualize(image_list, results, labels, output_dir='output/', threshold=0.86):
    # visualize the predict result
    start_idx = 0
    for idx, image_file in enumerate(image_list):
        im_bboxes_num = results['boxes_num'][idx]
        im_results = {}
        if 'boxes' in results:
            im_results['boxes'] = results['boxes'][start_idx:start_idx +
                                                   im_bboxes_num, :]
        if 'masks' in results:
            im_results['masks'] = results['masks'][start_idx:start_idx +
                                                   im_bboxes_num, :]
        if 'segm' in results:
            im_results['segm'] = results['segm'][start_idx:start_idx +
                                                 im_bboxes_num, :]
        if 'label' in results:
            im_results['label'] = results['label'][start_idx:start_idx +
                                                   im_bboxes_num]
        if 'score' in results:
            im_results['score'] = results['score'][start_idx:start_idx +
                                                   im_bboxes_num]

        start_idx += im_bboxes_num
        im = visualize_box_mask(
            image_file, im_results, labels, threshold=threshold)
        img_name = os.path.split(image_file)[-1]
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # out_path = os.path.join(output_dir, img_name)
        cv2_img = cv2.cvtColor(numpy.asarray(im),cv2.COLOR_RGB2BGR)
        cv2.imshow('res',cv2_img)
        cv2.waitKey(1)
        cv2.imwrite("rest/"+img_name,cv2_img)
        # im.save(out_path, quality=95)
        # print("save result to: " + out_path)


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


def predict_image_old(detector, image_list, batch_size=1):
    batch_loop_cnt = math.ceil(float(len(image_list)) / batch_size)
    for i in range(batch_loop_cnt):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(image_list))
        batch_image_list = image_list[start_index:end_index]
        if False:
            # FLAGS.run_benchmark:
            detector.predict(
                batch_image_list, FLAGS.threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
            print('Test iter {}'.format(i))
        else:
            t1= time.time()
            results = detector.predict(batch_image_list, FLAGS['threshold'])
            visualize(
                batch_image_list,
                results,
                detector.pred_config.labels,
                output_dir=FLAGS['output_dir'],
                threshold=FLAGS['threshold'])
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 
            print("Inference: {} ms per batch image".format(ms))

def predict_image(detector):
    # cv2.namedWindow('res', cv2.WINDOW_AUTOSIZE)
    global controler_tcpCliSock
    # results = detector.predict('output/001(1).jpg',0.5)
    while 1:
       
        if False: #读取文件夹内的图片
            dir_path='D:\\FTP Root\\ICR_EXT'
            path = Path(dir_path)
            file_names = path.glob('**/*.jpg')
            for target_list in file_names:
                
                if True: #imagelist.qsize() > 0:
                    # img_path = imagelist.get()     
                    t1=time.time();     
                    name=[os.path.join(dir_path, target_list.name)]           
                    results = detector.predict(name,FLAGS['threshold'])

                    visualize(
                        name,
                        results,
                        detector.pred_config.labels,
                        output_dir=FLAGS['output_dir'],
                        threshold=FLAGS['threshold'])
                    t2 = time.time()
                    ms = (t2 - t1) * 1000.0 / 1
                    print("Inference: {} ms per batch image".format(ms))
                    send_out={}
                    send_out['imag']=os.path.join(dir_path, target_list)
                    send_out['dettime']="{}".format(ms)
                    box_list=[]
                    if 'boxes' in results:                         
                        for box in results['boxes']:
                        #    box_res={}
                           if   box.size<1:
                                continue
                           if   box[1]>FLAGS['threshold'] :
                               box_res= str(box[0])+'|'+str(box[1])
                               box_list.append(box_res)                       
                    send_out['box']=  box_list
                    print(send_out)
                    j=json.dumps(send_out)
                    data=send_out
                    cv2.waitKey(-1)
                    # controler_tcpCliSock.send(bytes(j, 'utf8'))
        
        if True: #等待队列
            
            if handler.imagelist.qsize() > 0:
                tA = time.time()
                img_path = handler.imagelist.get()
                cv2.waitKey(5)

                results = detector.predict([img_path],FLAGS['threshold'])
                visualize(
                    [img_path],
                    results,
                    detector.pred_config.labels,
                    output_dir='output',
                    threshold=FLAGS['threshold'])
                t2 = time.time()
                ms = (t2 - tA) * 1000.0 / 1
                print("predict time: {} ms per batch image".format(ms))
                send_out={}
                send_out['imag']=img_path
                send_out['dettime']="{}".format(ms)
                box_list=[]
                if 'boxes' in results:                         
                    for box in results['boxes']:
                    #    box_res={}
                        if   box.size<1:
                            continue
                        if   box[1]>FLAGS['threshold'] :
                            box_res= str(box[0])+'|'+str(box[1])
                            box_list.append(box_res)                       
                send_out['box']=  box_list
                print(send_out)
                j=json.dumps(send_out)
                data=send_out
                controler_tcpCliSock.send(bytes(j, 'utf8'))
            else :
                time.sleep(0.001)
                continue
                


def main():
    pred_config = PredictConfig(FLAGS['model_dir'])
    detector_func = 'Detector'
    if pred_config.arch == 'SOLOv2':
        detector_func = 'DetectorSOLOv2'
    elif pred_config.arch == 'PicoDet':
        detector_func = 'DetectorPicoDet'

    detector = eval(detector_func)(pred_config,
                                   FLAGS['model_dir'],
                                   device=FLAGS['device'],
                                   run_mode=FLAGS['run_mode'],
                                   batch_size=FLAGS['batch_size'],
                                   trt_min_shape=FLAGS['trt_min_shape'],
                                   trt_max_shape=FLAGS['trt_max_shape'],
                                   trt_opt_shape=FLAGS['trt_opt_shape'],
                                   trt_calib_mode=FLAGS['trt_calib_mode'],
                                   cpu_threads=FLAGS['cpu_threads'],
                                   enable_mkldnn=FLAGS['enable_mkldnn'])

    # predict from image

    predict_image(detector)



def socket_threading(name):
    while True:
        conn, addr = s.accept()
        print("连接地址：", addr)
        client_list.append(conn)
        while True:

            try:
                data = conn.recv(1024)  # 接收数据
                if len(data)==0:
                    client_list.remove(conn)
                    break
                print('recive:', data.decode())  # 打印接收到的数据
                s_data = str(data, encoding="utf-8")
                imagelist.put(s_data)
                #conn.send(data.upper())  # 然后再发送数据
            except ConnectionResetError as e:
                client_list.remove(conn)
                print('关闭了正在占线的链接！')
                break
        conn.close()

def FTP_threading():
    # 开启服务器
    server = FTPServer(("0.0.0.0", 6000), handler)
    server.serve_forever()

def start_controler_Client():
    BUFSIZE = 1024
    ADDR = (HOST_controler, PORT_controler)
    global controler_tcpCliSock
    global controler_connected

    while True:
        controler_tcpCliSock = socket(AF_INET, SOCK_STREAM)
        port=random.randint(11560,11569)
        controler_tcpCliSock.settimeout(0.030)

        try:
        	controler_tcpCliSock.bind(('0.0.0.0', port))
        except Exception as e:
        	print(e)
        	time.sleep(3)
        	continue
        while True:
            try:
                controler_tcpCliSock.connect(ADDR)
                controler_connected=True
            except Exception as e:
                print(e)
                time.sleep(5)
                continue
            print("connectted to control..")
            break
        #break
        while controler_connected:
            try:
            	data = controler_tcpCliSock.recv(BUFSIZE).decode()
            	print("controler :"+data)
            	if len(data) == 0:
                	controler_tcpCliSock.close()
                	controler_tcpCliSock = socket(AF_INET, SOCK_STREAM)
                	print('Controler disconnected......')
                	break
            except Exception as e:
                    #print(e)
                    pass
        if controler_connected==False:
            controler_tcpCliSock.close()
            print('Controler disconnected......')
            # logger.warning('Controler disconnected......')


if __name__ == '__main__':
    paddle.enable_static()
    s = socket()
    s.bind(("127.0.0.1", 2114))
    s.listen(-1)

    client_list = []
    thread1 = Thread(target=socket_threading, args=("Thread-1",))
    thread1.start()

    thread2 = Thread(target=FTP_threading)
    thread2.start()

    thread3 = Thread(target=start_controler_Client)
    thread3.start()

    main()
