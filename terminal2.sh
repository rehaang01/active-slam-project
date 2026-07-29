# ──────────────────────────────────────────────
# STEP 7: Wait for simulation to be ready
# PX4 needs ~10s to boot, Gazebo needs ~5s to
# load the warehouse, RTAB-Map needs ~3s after
# that.  We wait until the key topics exist.
# ──────────────────────────────────────────────
echo -e "${YELLOW}[7/8] Waiting for simulation to be ready...${NC}"

wait_for_topic() {
    local topic=$1
    local label=$2
    local timeout=$3
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if ros2 topic list 2>/dev/null | grep -q "$topic"; then
            echo -e "  ${GREEN}✓ $label${NC}"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "    Still waiting for $label... (${elapsed}s)"
        fi
    done
    echo -e "  ${RED}✗ $label (timeout after ${timeout}s)${NC}"
    return 1
}

echo "  Waiting up to 60s for PX4..."
wait_for_topic "/fmu/out/vehicle_local_position" "PX4 position data" 60

echo "  Waiting up to 60s for RTAB-Map..."
wait_for_topic "/rtabmap/octomap_grid" "RTAB-Map occupancy grid" 60

echo "  Waiting up to 30s for LiDAR..."
wait_for_topic "/scan" "LiDAR scan" 30

# ──────────────────────────────────────────────
# STEP 8: Verify sensors are actually publishing
# Topics existing ≠ topics publishing data.
# Check that each one has at least one message.
# ──────────────────────────────────────────────
echo -e "${YELLOW}[8/8] Verifying sensor data flow...${NC}"

verify_topic() {
    local topic=$1
    local label=$2
    local timeout=10
    # Try to get one message with timeout
    if timeout $timeout ros2 topic echo "$topic" --once >/dev/null 2>&1; then
        local hz=$(ros2 topic hz "$topic" --window 5 2>/dev/null &
            sleep 3
            kill %1 2>/dev/null
        )
        echo -e "  ${GREEN}✓ $label — publishing${NC}"
        return 0
    else
        echo -e "  ${RED}✗ $label — NOT publishing (check bridge/sensor)${NC}"
        return 1
    fi
}

ALL_OK=true
verify_topic "/fmu/out/vehicle_local_position" "PX4 position" || ALL_OK=false
verify_topic "/scan" "LiDAR" || ALL_OK=false
verify_topic "/rtabmap/odom" "RTAB-Map odometry" || ALL_OK=false

echo ""
echo "============================================"
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN} ✓ ALL SYSTEMS READY${NC}"
else
    echo -e "${YELLOW} ⚠ Some sensors not verified — check above.${NC}"
    echo "  The simulation may still be starting up."
    echo "  Wait 30s and re-check with:"
    echo "    ros2 topic echo /scan --once"
fi
echo ""
echo " SIMULATION IS RUNNING."
echo ""
echo " Now run:"
echo "   sudo docker exec -it rtabmap_ros2 bash"
echo "   source /opt/ros/humble/setup.bash"
echo "   cd /root/active_slam_project"
echo ""
echo " Then run ONE of:"
echo "   python3 baseline_random_walk.py      # ~66 min"
echo "   python3 baseline_frontier.py         # ~66 min"
echo "   python3 baseline_spiral.py           # ~66 min"
echo "   python3 baseline_potential_field.py  # ~66 min"
echo "   python3 train.py                     # ~24 hours"
echo ""
echo " IMPORTANT:"
echo "   • Restart simulation between each run:"
echo "     tmuxinator stop rtabmap_ros2_gazebo"
echo "     tmuxinator start rtabmap_ros2_gazebo"
echo "     (wait 30s for sensors)"
echo ""
echo "   • Monitor training with TensorBoard:"
echo "     tensorboard --logdir logs/tb --host 0.0.0.0"
echo ""
echo " This terminal will stay here. DO NOT close it."
echo " The simulation runs in tmux in the background."
echo "============================================"