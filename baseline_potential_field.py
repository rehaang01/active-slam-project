#!/usr/bin/env python3
"""Gazebo potential-field baseline on the shared 3D observation/action contract."""

from gazebo_baseline_runner import run_gazebo_baseline
from lightweight_train.baselines_3d import PotentialField3D


def main():
    run_gazebo_baseline(PotentialField3D(), prefix="baseline_potential_field", title="Potential Field")


if __name__ == "__main__":
    main()
