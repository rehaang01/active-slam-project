#!/usr/bin/env python3
"""Shared transfer-compatible feature extractor for Gazebo training."""

from shared_slam_policy import CoordConv2d, TransferSLAMFeatureExtractor as SLAMFeatureExtractor

__all__ = ["CoordConv2d", "SLAMFeatureExtractor"]
