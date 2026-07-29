#!/usr/bin/env python3
"""Gazebo nearest-frontier baseline on the shared 3D observation/action contract."""

from gazebo_baseline_runner import run_gazebo_baseline
from lightweight_train.baselines_3d import NearestFrontier3D


def main():
    run_gazebo_baseline(NearestFrontier3D(), prefix="baseline_frontier", title="Nearest Frontier")


if __name__ == "__main__":
    main()
