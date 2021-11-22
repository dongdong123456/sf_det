import socket
from socket import *
from threading import Thread
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import random
import argparse
import os
import time
from loguru import logger
import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
logger.add('{time}.log', rotation='00:00',retention="10 days")
# 新建一个用户组
authorizer = DummyAuthorizer()
# 将用户名，密码，指定目录，权限 添加到里面
if not os.path.exists("F:\\FTP Root\\ICR_EXT"):
    os.mkdir("F:\\FTP Root\\ICR_EXT")
authorizer.add_user("sick", "sick", "F:\\FTP Root\\ICR_EXT", perm="elradfmw")  # adfmw
handler = FTPHandler
handler.authorizer = authorizer
handler.passive_ports = range(5001, 5510)



controler_tcpCliSock = socket(AF_INET, SOCK_STREAM)
HOST_controler = '192.168.100.101'  # The remote host
PORT_controler = 3500  # The same port as used by the server

# imagelist = queue.Queue()

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="best_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.85, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser

def image_demo(predictor, vis_folder, path, current_time, save_result):

    img_path = handler.imagelist.get()
    time.sleep(0.01)
    t1 = time.time()
    outputs, img_info = predictor.inference(img_path)
    # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    t2 = time.time()
    ms = (t2 - t1) * 1000.0 / 1
    logger.info("predict image_name: {} ".format(img_info['file_name']))
    logger.info("predict time: {} ms one image".format(ms))
    #print("predict time: {} ms per batch image".format(ms))
    Track_id=img_info['file_name'].split('_')[6]
    label=1
    if len(outputs)>0:
        if outputs[0] is not None :
            if  outputs[0].shape[0]>1:
                picked_boxes= nms(outputs[0])
                if len(picked_boxes)>1:
                    label = 2
            output=outputs[0]
            bboxes=output[:,0:4]
            cls=output[:,6]
            scores=output[:,4]*output[:,5]			
            logger.info("predict bboxes: {} ".format(bboxes))
            logger.info("predict scores: {} ".format(scores))
            logger.info("predict result: {} ".format(label))
    send_out='\u0002'+str(Track_id)+';'+str(label)+'\u0003'
    controler_tcpCliSock.send(bytes(send_out, 'utf8'))
    # save_file_name = img_path.split("_EEEEEEEEEEEE_")[-1]
    # img_path="F:/my_code/2021/sf/1027/"+save_file_name
    # # print("img_name:{}".format(save_file_name))
    # #     logger.info("Saving detection result in {}".format(save_file_name))
    # cv2.imwrite(img_path, result_image)
    # return result_image


def nms(bboxes):
    """非极大抑制过程
    :param bboxes: 同类别候选框坐标
    :return:
    """

    bboxes = np.array(bboxes.cpu())
    # 取出n个的极坐标点
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4] * bboxes[:, 5]
    # 2、对候选框进行NMS筛选
    # 从大到小排列，取index
    order = scores.argsort()[::-1]
    areas = (x2 - x1) * (y2 - y1)
    # 2、对候选框进行NMS筛选
    # 返回的框坐标和分数
    picked_boxes = []
    while order.size > 0:
        # 将当前置信度最大的框加入返回值列表中
        index = order[-1]
        picked_boxes.append(bboxes[index])

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])

        # 计算当前矩形框与其余框的比值
        rate = areas[index] / areas[order[:-1]]
        # 计算其余框于u当前框的比值
        rate1 = areas[order[:-1]] / areas[index]

        w = np.maximum(0.0, x22 - x11)
        h = np.maximum(0.0, y22 - y11)
        intersection = w * h

        # 利用相交的面积和两个框自身的面积计算框的交并比, 保留大于阈值的框
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        # rate==ratio表示包含关系，保留不为包含关系的框
        keep_boxes_indics = np.where(abs(ratio - rate)>0.01)
        keep_boxes_indics1 = np.where(abs(ratio - rate1)>0.01)
        if keep_boxes_indics[0].__len__() < keep_boxes_indics1[0].__len__():
            order = order[keep_boxes_indics]
        else:
            order = order[keep_boxes_indics1]


    return picked_boxes

    """非极大抑制过程
    :param bboxes: 同类别候选框坐标
    :return:
    """

    bboxes = np.array(bboxes.cpu())
    # 取出n个的极坐标点
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # 2、对候选框进行NMS筛选
    # 返回的框坐标和分数
    picked_boxes = []
    # 对置信度进行排序, 获取排序后的下标序号, argsort默认从小到大排序
    order = np.argsort(np.ones(len(bboxes)))
    areas = (x2 - x1) * (y2 - y1)
    while order.size > 1:
        # 将当前置信度最大的框加入返回值列表中
        index = order[-1]
        picked_boxes.append(bboxes[index])

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])

        # 计算当前矩形框与其余框的比值
        rate = areas[index] / areas[order[:-1]]
        # 计算其余框于u当前框的比值
        rate1 = areas[order[:-1]] / areas[index]

        w = np.maximum(0.0, x22 - x11)
        h = np.maximum(0.0, y22 - y11)
        intersection = w * h

        # 利用相交的面积和两个框自身的面积计算框的交并比, 保留大于阈值的框
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        # rate==ratio表示包含关系，保留不为包含关系的框
        keep_boxes_indics = np.where(ratio != rate)
        keep_boxes_indics1 = np.where(ratio != rate1)

        if keep_boxes_indics.__len__() < keep_boxes_indics1.__len__():
            order = order[keep_boxes_indics]
        else:
            order = order[keep_boxes_indics1]
    return picked_boxes



class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None


        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)
    current_time = time.localtime()
    while True:
        if handler.imagelist.qsize() > 0:
            # result_image=image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
            try:
                image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
            except Exception:
                #logger.error("Exception :{}").format(Exception)
                continue
            # cv2.imshow("show", result_image)
            # cv2.waitKey(1)
        else:
            time.sleep(0.0001)
            continue




# def socket_threading():
#     while True:
#         conn, addr = s.accept()
#         print("连接地址：", addr)
#         client_list.append(conn)
#         while True:
#             try:
#                 data = conn.recv(1024)  # 接收数据
#                 if len(data) == 0:
#                     client_list.remove(conn)
#                     break
#                 print('recive:', data.decode())  # 打印接收到的数据
#                 s_data = str(data, encoding="utf-8")
#                 imagelist.put(s_data)
#                 # conn.send(data.upper())  # 然后再发送数据
#             except ConnectionResetError as e:
#                 client_list.remove(conn)
#                 print('关闭了正在占线的链接！')
#                 break
#         conn.close()


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
        port = random.randint(11560, 11569)
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
                controler_connected = True
            except Exception as e:
                print(e)
                time.sleep(5)
                continue
            print("connectted to control..")
            break
        # break
        while controler_connected:
            try:
                data = controler_tcpCliSock.recv(BUFSIZE).decode()
                print("controler :" + data)
                if len(data) == 0:
                    controler_tcpCliSock.close()
                    controler_tcpCliSock = socket(AF_INET, SOCK_STREAM)
                    print('Controler disconnected......')
                    break
            except Exception as e:
                # print(e)
                pass
        if controler_connected == False:
            controler_tcpCliSock.close()
            print('Controler disconnected......')
            # logger.warning('Controler disconnected......')




if __name__ == '__main__':
    # s = socket()
    # s.bind(("127.0.0.1", 2114))
    # s.listen(-1)
    #
    # client_list = []
    # thread1 = Thread(target=socket_threading)
    # thread1.start()

    thread2 = Thread(target=FTP_threading)
    thread2.start()

    thread3 = Thread(target=start_controler_Client)
    thread3.start()

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

