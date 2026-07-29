#!/usr/bin/env python3
"""Gazebo random-walk baseline on the shared 3D observation/action contract."""

from gazebo_baseline_runner import run_gazebo_baseline
from lightweight_train.baselines_3d import RandomWalk3D


def main():
    run_gazebo_baseline(RandomWalk3D(), prefix="baseline_random_walk", title="Random Walk")


if __name__ == "__main__":
    main()
