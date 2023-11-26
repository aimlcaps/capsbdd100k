import argparse
import math
import os
import random

import cv2
import numpy as np
import yaml
from torch import nn
from google.colab.patches import cv2_imshow

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from models.yolo import Model
import torch
from pathlib import Path

class Model(nn.ModuleList):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)  # nms ensemble
        return y
def loadModel(opt,hyp,nc, wf,device):
    model = Model()
    weights = torch.load(wf, map_location=device)  # load
    model = weights['ema' if weights.get('ema') else 'model'].float()
    return model

def evaluate(opt,hyp,nc,classes,wf,path,img,device):
    model = loadModel(opt,hyp,nc,wf,device)
    model.eval()
    stride = int(model.stride.max())
    imgsize = math.ceil(opt.img_size / stride) * stride #Image size should be multiples of stride
    dataset = LoadImages(path+"//"+img, img_size=imgsize, stride=stride)
    half = False #device.type != 'cpu'
    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]

        # Apply NMS
        final_pred = non_max_suppression(pred, float(opt.conf_thres), float(opt.iou_thres), classes=classes)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Process detections
        for i, det in enumerate(final_pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(path+"//evaluated_"+p.name)  # img.jpg
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            #cv2.imshow(str(p), im0)
            cv2_imshow(im0)
            cv2.waitKey(10)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)

class LoadImages:
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())
        files = [p]  # files

        images = [x for x in files]
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni  # number of files
        self.mode = 'image'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0

    def __len__(self):
        return self.nf  # number of files


if __name__ == '__main__':
    OPT_FILE_PATH ='S:\IIITH\Capstone-ImageTagging -GoogleDrive\Capstone-ImageTagging\config\opt.yaml'

    HYP_FILE_PATH='S:\IIITH\Capstone-ImageTagging -GoogleDrive\Capstone-ImageTagging\config\hyp.yaml'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(OPT_FILE_PATH) as f:
      opt_params = yaml.load(f, Loader=yaml.SafeLoader)

    with open(HYP_FILE_PATH) as f:
      hyp_params = yaml.load(f, Loader=yaml.SafeLoader)

    parser = argparse.ArgumentParser()
    for (key,value) in opt_params.items():
      parser.add_argument('--'+key,default=value)

    opt_dict = parser.parse_args(args=[])

    weights = 'S:\IIITH\Capstone-ImageTagging -GoogleDrive\Capstone-ImageTagging\pycode\yolov7\\runs\\train\exp8\weights\\best.pt'
    #names = ["car", "bus", "person", "bike", "truck", "motor", "train", "rider", "traffic sign", "traffic light"]
    class_indicies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_img_path = 'S:\IIITH\Capstone-ImageTagging -GoogleDrive\Capstone-ImageTagging\dataset\\bdd100k\images\\test'
    test_img = random.choice(os.listdir(test_img_path))
    evaluate(opt_dict,hyp_params,10,class_indicies,weights,test_img_path,test_img, device)