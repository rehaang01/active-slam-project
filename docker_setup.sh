#!/bin/bash
# ============================================================
# master_setup.sh — Complete setup + launch for Active SLAM
#
# Run this in Terminal 1 INSIDE the Docker container.
# It does everything in the correct order:
#   1. Sources ROS2
#   2. Applies LiDAR mass fix (critical for altitude stability)
#   3. Adds LiDAR to drone model + installs launch file
#   4. Installs Python dependencies (correct order)
#   5. Verifies CUDA
#   6. Starts simulation (PX4 + RTAB-Map + bridges)
#   7. Waits for simulation to be ready
#   8. Verifies all sensors
#
# Usage (Terminal 1, inside Docker):
#   cd /root/active_slam_project
#   bash master_setup.sh
#
# Then in Terminal 2:
#   sudo docker exec -it rtabmap_ros2 bash
#   source /opt/ros/humble/setup.bash
#   cd /root/active_slam_project
#   python3 baseline_random_walk.py   # or train.py
# ============================================================

set -e  # Exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================"
echo " Active SLAM — Master Setup"
echo "============================================"
echo ""

# ──────────────────────────────────────────────
# STEP 1: Source ROS2
# Without this, ros2 commands, tmuxinator launch,
# and topic checks all fail silently.
# ──────────────────────────────────────────────
echo -e "${YELLOW}[1/8] Sourcing ROS2 Humble...${NC}"
source /opt/ros/humble/setup.bash
echo -e "${GREEN}  Done.${NC}"

# ──────────────────────────────────────────────
# STEP 2: Apply LiDAR mass fix (CRITICAL)
# The lidar_2d_v2 model adds 0.37kg to the drone.
# PX4's altitude controller can't handle this
# during lateral movement → drone descends and
# crashes.  Reducing to 0.05kg fixes it.
# ──────────────────────────────────────────────
echo -e "${YELLOW}[2/8] Applying LiDAR mass fix...${NC}"
LIDAR_MODEL="/root/PX4-Autopilot/Tools/simulation/gz/models/lidar_2d_v2/model.sdf"
if [ -f "$LIDAR_MODEL" ]; then
    CURRENT_MASS=$(grep -oP '<mass>\K[^<]+' "$LIDAR_MODEL" | head -1)
    if [ "$CURRENT_MASS" = "0.05" ]; then
        echo -e "${GREEN}  Already fixed (mass=0.05kg). Skipping.${NC}"
    else
        sed -i 's/<mass>0.37<\/mass>/<mass>0.05<\/mass>/' "$LIDAR_MODEL"
        # Also handle case where mass was already changed to something else
        sed -i 's/<mass>[0-9.]*<\/mass>/<mass>0.05<\/mass>/' "$LIDAR_MODEL"
        NEW_MASS=$(grep -oP '<mass>\K[^<]+' "$LIDAR_MODEL" | head -1)
        echo -e "${GREEN}  Fixed: ${CURRENT_MASS}kg → ${NEW_MASS}kg${NC}"
    fi
else
    echo -e "${RED}  WARNING: LiDAR model not found at $LIDAR_MODEL${NC}"
    echo -e "${RED}  Altitude stability may be compromised.${NC}"
fi

# ──────────────────────────────────────────────
# STEP 3: Add LiDAR to drone model + install launch file
# Same as docker_setup.sh steps 1 and 2, but
# without the pip installs (we do those separately
# in the correct order).
# ──────────────────────────────────────────────
echo -e "${YELLOW}[3/8] Setting up drone model + launch file...${NC}"
MODEL_SDF="/root/PX4-Autopilot/Tools/simulation/gz/models/x500_depth/model.sdf"

if grep -q "lidar_2d_v2" "$MODEL_SDF" 2>/dev/null; then
    echo "  LiDAR already in drone model. Skipping."
else
    echo "  Adding LiDAR to x500_depth model..."
    cp "$MODEL_SDF" "${MODEL_SDF}.backup"
    cp /root/active_slam_project/model.sdf "$MODEL_SDF"
    echo "  Done."
fi

LAUNCH_DIR=$(ros2 pkg prefix simulation_launch 2>/dev/null)/share/simulation_launch/launch
if [ -d "$LAUNCH_DIR" ]; then
    cp /root/active_slam_project/launch/full_simulation.launch.py "$LAUNCH_DIR/"
    echo -e "${GREEN}  Launch file installed to $LAUNCH_DIR${NC}"
else
    echo -e "${RED}  WARNING: simulation_launch package not found.${NC}"
    echo -e "${RED}  Make sure ROS2 is sourced (Step 1).${NC}"
    exit 1
fi

# ──────────────────────────────────────────────
# STEP 4: Install Python dependencies
# Order matters:
#   1. Base packages (protobuf, gymnasium, sb3)
#   2. PyTorch with CUDA (force-reinstall to get
#      correct CUDA-linked version)
#   3. numpy<2.0 LAST (PyTorch may install numpy 2.x
#      which breaks ROS2 message serialization)
# ──────────────────────────────────────────────
echo -e "${YELLOW}[4/8] Installing Python dependencies...${NC}"
echo "  Installing base packages..."
pip3 install -q protobuf==3.20.1 2>/dev/null
pip3 install -q stable-baselines3[extra] sb3-contrib gymnasium 2>/dev/null
echo "  Installing PyTorch with CUDA..."
pip3 install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall 2>/dev/null
echo "  Pinning numpy<2.0 (ROS2 compatibility)..."
pip3 install -q "numpy<2.0" --force-reinstall 2>/dev/null
echo -e "${GREEN}  Done.${NC}"

# ──────────────────────────────────────────────
# STEP 5: Verify CUDA
# ──────────────────────────────────────────────
echo -e "${YELLOW}[5/8] Verifying CUDA...${NC}"
CUDA_OK=$(python3 -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null)
if [ "$CUDA_OK" = "YES" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}  CUDA available: $GPU_NAME${NC}"
else
    echo -e "${RED}  WARNING: CUDA NOT available! Training will be ~10x slower.${NC}"
    echo -e "${RED}  Check: nvidia-smi, GPU passthrough, PyTorch CUDA version.${NC}"
fi