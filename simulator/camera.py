# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera configs."""

import numpy as np
import pybullet as p


class RealSenseD415:
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450.0, 0, 320.0, 0, 450.0, 240.0, 0, 0, 1)
    # intrinsics = (540., 0, 320., 0, 540., 240., 0, 0, 1)

    # Set default camera poses.
    front_position = (0.55, 0.15, 0.25)
    front_rotation = (np.pi / 2.5, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (-0.16, 0.43, 0.25)
    left_rotation = (np.pi / 2.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (-0.16, -0.12, 0.25)
    right_rotation = (np.pi / 2.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)
    # Default camera configs.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 2.0),
            "noise": False,
            "name": "front",
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": left_position,
            "rotation": left_rotation,
            "zrange": (0.01, 2.0),
            "noise": False,
            "name": "left",
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": right_position,
            "rotation": right_rotation,
            "zrange": (0.01, 2.0),
            "noise": False,
            "name": "right",
        },
    ]
