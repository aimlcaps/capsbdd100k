from PIL import Image
import random
import torch
import argparse
import logging
import math
import os, sys
from pathlib import Path
import numpy as np
import torch.optim as optim
import yaml
from torch.cuda import amp
from tqdm import tqdm

from models.yolo import Model
from utils.general import labels_to_class_weights,labels_to_image_weights, init_seeds, check_img_size,non_max_suppression,scale_coords,xywh2xyxy,box_iou
from utils.metrics import ap_per_class
from utils.loss import ComputeLoss

def val(val_dataloader,model,criterian,opt_dict):
  loss = torch.zeros(3, device=device)
  jdict, stats, ap, ap_class = [], [], [], []
  iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
  niou = iouv.numel()
  half = False
  for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(val_dataloader, desc=s)):
    img = img.to(device, non_blocking=True)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    targets = targets.to(device)
    nb, _, height, width = img.shape  # batch size, channels, height, width
    train_out = []
    with torch.no_grad():
        out = model(img)  # inference and training outputs
        loss += criterian([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = []#targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        out = non_max_suppression(out, conf_thres=opt_dict.conf_thres, iou_thres=opt_dict.iou_thres, labels=lb, multi_label=True)
    # Statistics per image
    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        path = Path(paths[si])
        seen += 1

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
            # if plots:
            #     confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    # plot_images(img, targets, paths, f, names)
    # plot_images(img, output_to_target(out), paths, f, names)
  stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
  if len(stats) and stats[0].any():
      p, r, ap, f1, ap_class = ap_per_class(*stats, names=CLASS_NAMES)
      ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
      mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
      nt = np.bincount(stats[3].astype(np.int64), minlength=NUM_CLASSES)  # number of targets per class
  else:
      nt = torch.zeros(1)

  # Print results
  #pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
  #print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
  return mp,mr,map