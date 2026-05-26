#!/usr/bin/env bash
# Convert Alicia_D_v5_6 URDF -> USD for IsaacLab.
# Run from any directory; absolute paths are baked in.
set -e

# Activate conda env that has IsaacLab installed
source /home/spy/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

ISAACLAB_DIR="/home/spy/graduate_student/quadruped_robot/Reinforcement_Learning/IsaacLab"
PROJECT_DIR="/home/spy/graduate_student/quadruped_robot/Reinforcement_Learning/mechanical arm/Isaaclab_alicia"
URDF="$PROJECT_DIR/source/Isaaclab_alicia/Isaaclab_alicia/assets/alicia_duo/urdf/Alicia_D_v5_6_gripper_100mm.urdf"
USD="$PROJECT_DIR/source/Isaaclab_alicia/Isaaclab_alicia/assets/alicia_duo/usd/alicia_duo.usd"

mkdir -p "$(dirname "$USD")"

cd "$ISAACLAB_DIR"
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    "$URDF" \
    "$USD" \
    --merge-joints \
    --fix-base \
    --joint-stiffness 0.0 \
    --joint-damping 0.0 \
    --joint-target-type none \
    --headless

echo ""
echo "[OK] USD written to: $USD"
ls -la "$USD"
