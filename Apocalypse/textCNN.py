#本模型是利用textCNN来做的一个特征提取模型，用的anaconda的tensorflow环境下的pytorch1.5
import os
import torch
from torch import nn
import torch.utils.data as Data
import  torch.nn.functional as F
import sys