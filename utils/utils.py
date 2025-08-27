import torch
import numpy as np
from medpy import metric
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hausdorff import hausdorff_distance
from scipy.spatial.distance import cdist
from scipy.spatial.distance import directed_hausdorff


def calculate_metric_percase(y_true, y_pred, thr=0.5):
    y_true = y_true.to(torch.float32).cpu().detach().numpy().astype(bool)
    y_pred = (y_pred > thr).astype(bool)

    if y_pred.sum() == 0:
        dice = 0
        jc = 0
        hd = 100
        asd = 100
    else:
        dice = metric.binary.dc(y_pred, y_true)
        jc = metric.binary.jc(y_pred, y_true)
        hd = metric.binary.hd95(y_pred, y_true)
        asd = metric.binary.asd(y_pred, y_true)
    return dice, jc, hd, asd



def patients_to_slices(dataset, patiens_num):
    ref_dict = {}
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "tumor" in dataset:
        ref_dict = {"30": 435}
    elif "ISIC" in dataset:
        ref_dict = {"30": 622}
    elif "LA_Seg_Training" in dataset:
        ref_dict = {"5": 4,"10": 8}
    elif "Pancreas" in dataset:
        ref_dict = {"5": 3, "10": 6, "20": 12, "30": 18}

    elif "BraTS" in dataset:
        ref_dict = {"5": 12,"10": 25}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


