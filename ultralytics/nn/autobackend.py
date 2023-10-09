# Ultralytics YOLO 🚀, AGPL-3.0 license

import ast
import contextlib
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils.ops import xywh2xyxy


class AutoBackend(nn.Module):

    def __init__(self,
                 weights='yolov8n.pt',
                 device=torch.device('cpu'),
                 dnn=False,
                 data=None,
                 fp16=False,
                 fuse=True,
                 verbose=True):
        
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        pt = True
        fp16 &= pt 
        nhwc = False
        stride = 32  # default stride
        model, metadata = None, None
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        if nn_module:
            model = weights.to(device)
            model = model.fuse(verbose=verbose) if fuse else model
            if hasattr(model, 'kpt_shape'):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True
        if pt:  # PyTorch
            from ultralytics.nn.tasks import attempt_load_weights
            model = attempt_load_weights(weights if isinstance(weights, list) else w,
                                         device=device,
                                         inplace=True,
                                         fuse=fuse)
            if hasattr(model, 'kpt_shape'):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata:
            for k, v in metadata.items():
                if k in ('stride', 'batch'):
                    metadata[k] = int(v)
                elif k in ('imgsz', 'names', 'kpt_shape') and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata['stride']
            task = metadata['task']
            batch = metadata['batch']
            imgsz = metadata['imgsz']
            names = metadata['names']
            kpt_shape = metadata.get('kpt_shape')

        # Check names
        if 'names' not in locals():  # names missing
            names = self._apply_default_class_names(data)
        #names = check_class_names(names)

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt or self.nn_module:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    @staticmethod
    def _apply_default_class_names(data):
        """Applies default class names to an input YAML file or returns numerical class names."""
        with contextlib.suppress(Exception):
            return yaml_load(data)['names']
        return {i: f'class{i}' for i in range(999)}  # return default if above errors

