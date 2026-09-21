# coding: utf-8

"""
config dataclass used for inference
"""

import cv2
from numpy import ndarray
import pickle as pkl
from dataclasses import dataclass, field
from typing import Tuple
from .base_config import PrintableConfig, make_abs_path

def load_lip_array():
    with open(make_abs_path('../utils/resources/lip_array.pkl'), 'rb') as f:
        return pkl.load(f)

@dataclass(repr=False)  # use repr from PrintableConfig
class InferenceConfig(PrintableConfig):
    # HUMAN MODEL CONFIG, NOT EXPORTED PARAMS
    models_config: str = make_abs_path('./models.yaml')  # portrait animation config
    checkpoint_F: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth')  # path to checkpoint of F
    checkpoint_M: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/motion_extractor.pth')  # path to checkpoint pf M
    checkpoint_G: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/spade_generator.pth')  # path to checkpoint of G
    checkpoint_W: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/warping_module.pth')  # path to checkpoint of W
    checkpoint_S: str = make_abs_path('../../pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth')  # path to checkpoint to S and R_eyes, R_lip

    flag_use_half_precision: bool = True
    device_id: int = 0
    flag_normalize_lip: bool = True
    flag_stitching: bool = True
    flag_relative_motion: bool = True
    flag_do_rot: bool = True
    flag_force_cpu: bool = False
    flag_do_torch_compile: bool = False
    driving_multiplier: float = 1.0
    source_max_dim: int = 1280
    source_division: int = 2

    lip_normalize_threshold: float = 0.03
    input_shape: Tuple[int, int] = (256, 256)

    mask_crop: ndarray = field(default_factory=lambda: cv2.imread(make_abs_path('../utils/resources/mask_template.png'), cv2.IMREAD_COLOR))
    lip_array: ndarray = field(default_factory=load_lip_array)
