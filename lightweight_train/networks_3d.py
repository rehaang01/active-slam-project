"""Shared 3D SLAM feature extractors re-exported for lightweight training."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from shared_slam_policy import CoordConv2d, TransferSLAMFeatureExtractor as SLAMFeatureExtractor3D

__all__ = ["CoordConv2d", "SLAMFeatureExtractor3D"]
