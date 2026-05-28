# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments (optional outside Isaac Sim runtime).
try:
    from .tasks import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

# Register UI extensions (optional in deployment-only environments).
try:
    from .ui_extension_example import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass
