#!/usr/bin/env python3
"""
SLAMDataCollector: ROS2 node that bridges RTAB-Map/PX4 data to the RL environment.

Architecture alignment:
- /rtabmap/localization_pose → global covariance (core novelty signal)
- /rtabmap/octomap_grid      → 2D Z-sliced occupancy grid (CNN Channel 1)
- /rtabmap/octomap_full       → 3D OctoMap (volumetric info gain in reward)
- /scan                       → 2D LiDAR (collision detection + reactive safety)
- /depth_camera               → depth image (close-range reactive avoidance)
- set_z_filter()              → dynamic altitude slicing for projected map
- PX4 topics                  → drone position, status, and flight commands

Key design decisions:
- VOLATILE QoS for PX4 topics (matches PX4's DDS profile)
- RELIABLE QoS for SLAM topics (matches RTAB-Map defaults)
- BEST_EFFORT for sensor topics (LiDAR, depth) for lowest latency
- Z-filter state tracked so env can query current slice altitude
- LiDAR data stored as numpy array with angular metadata for directional queries
- Tracking loss detection via odometry timeout (not loop-closure counting)
- Map NEVER resets — persists and grows across episodes

Fixes applied (v2):
  - FIX 1: get_lidar_range_in_direction now converts world-frame angles to
            body-frame automatically (was comparing world yaw against body-frame
            LiDAR angles — looked at wrong sector entirely)
  - FIX 2: Tracking loss detection rewritten — now based on odometry message
            timeout (1.5s) instead of counting frames without loop closures
            (which are normal and caused constant false positives)
  - FIX 3: Thread-safety locks on map and LiDAR state (callbacks run on
            executor threads, main thread reads same fields in step())
  - FIX 4: set_z_filter now waits for and checks the async service result
            (was fire-and-forget, recorded success without confirmation)
  - FIX 5: Covariance trace now includes full 6-DOF diagonal (was XYZ only,
            missed yaw uncertainty which matters for SLAM quality)
  - FIX 6: Dead frontier PointCloud2 code documented (kept for future use)
"""

import time as _time
import threading
import numpy as np
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2, LaserScan, Image
from rtabmap_msgs.msg import Info
from octomap_msgs.msg import Octomap

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


class SLAMDataCollector(Node):
    """Collects all SLAM, sensor, and PX4 data asynchronously for the RL environment."""

    def __init__(self):
        super().__init__("slam_data_collector")

        # === QoS Profiles ===
        # RELIABLE for SLAM topics (RTAB-Map uses RELIABLE by default)
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # BEST_EFFORT + VOLATILE for PX4 topics (matches PX4's QoS)
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # BEST_EFFORT for high-frequency sensor topics (LiDAR, depth)
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ============================================
        # SLAM Subscriptions
        # ============================================
        self.odom_sub = self.create_subscription(
            Odometry, "/rtabmap/odom", self._odom_cb, qos_reliable)

        self.loc_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/rtabmap/localization_pose",
            self._loc_pose_cb, qos_reliable)

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/rtabmap/octomap_grid",
            self._map_cb, qos_reliable)

        self.octomap_sub = self.create_subscription(
            Octomap, "/rtabmap/octomap_full",
            self._octomap_cb, qos_reliable)

        # NOTE (FIX 6): This subscription feeds get_frontier_points_2d() which
        # is currently unused — the env and baselines do their own BFS frontier
        # detection on the occupancy grid instead.  Kept for potential future
        # use (RTAB-Map's native frontiers are 3D-aware and faster than BFS).
        self.frontier_sub = self.create_subscription(
            PointCloud2, "/rtabmap/octomap_global_frontier_space",
            self._frontier_cb, qos_reliable)

        self.info_sub = self.create_subscription(
            Info, "/rtabmap/info",
            self._info_cb, qos_reliable)

        # ============================================
        # Sensor Subscriptions (LiDAR + Depth Camera)
        # ============================================
        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan",
            self._lidar_cb, qos_sensor)

        self.depth_sub = self.create_subscription(
            Image, "/depth_camera",
            self._depth_cb, qos_sensor)

        # ============================================
        # PX4 Subscriptions
        # ============================================
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self._local_pos_cb, qos_px4)
        self.status_sub = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._status_cb, qos_px4)

        # ============================================
        # PX4 Publishers
        # ============================================
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_px4)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_px4)
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_px4)

        # ============================================
        # Thread-safety locks (FIX 3)
        # Callbacks run on MultiThreadedExecutor threads while main
        # thread reads these fields in step().  Locks prevent torn
        # reads (e.g. new grid_info paired with old occupancy_grid).
        # ============================================
        self._map_lock = threading.Lock()
        self._lidar_lock = threading.Lock()
        self._covariance_lock = threading.Lock()

        # ============================================
        # SLAM Data Storage
        # ============================================
        self.odom = None
        self.occupancy_grid = None
        self.grid_info = None
        self.global_pose_covariance = np.zeros(36, dtype=np.float64)
        self.frontier_count = 0
        self.frontier_points = None
        self.loop_closure_id = 0
        self.proximity_detection_id = 0
        self.octomap_data = None

        # Map tracking
        self.known_cells_count = 0
        self.total_cells = 0

        # ============================================
        # Tracking quality (FIX 2 — rewritten)
        # Old approach counted consecutive frames without loop closures,
        # but loop_closure_id == 0 is the NORMAL state (loop closures are
        # rare).  The counter hit 45 within ~1.5s of normal exploration
        # and falsely declared tracking lost.
        #
        # New approach: monitor whether odometry messages keep arriving.
        # When RTAB-Map truly loses tracking, it stops publishing odom.
        # This is an unambiguous, reliable signal.
        # ============================================
        self.tracking_lost = False
        self._last_odom_time = None      # Set on first odom callback
        self._last_loc_pose_time = None  # Set on first localization pose
        self._tracking_lost_timeout = 1.5  # seconds without odom → lost

        # ============================================
        # LiDAR Data Storage
        # ============================================
        self.lidar_ranges = None
        self.lidar_min_range = float('inf')
        self.lidar_angle_min = 0.0
        self.lidar_angle_max = 0.0
        self.lidar_angle_increment = 0.0
        self.lidar_range_min = 0.0
        self.lidar_range_max = 0.0

        # ============================================
        # Depth Camera Data Storage
        # ============================================
        self.depth_min_distance = float('inf')
        self.depth_image = None

        # ============================================
        # PX4 Data Storage
        # ============================================
        self.local_position = None
        self.vehicle_status = None

        # ============================================
        # Z-filter State (for altitude-sliced maps)
        # ============================================
        self.set_params_client = self.create_client(
            SetParameters, "/rtabmap/rtabmap/set_parameters")
        self.z_filter_min = 0.0
        self.z_filter_max = 0.0
        self.z_filter_active = False

        self.get_logger().info("SLAM data collector initialized (v2 — fixed)")
        self.get_logger().info("  Covariance: /rtabmap/localization_pose (6-DOF)")
        self.get_logger().info("  3D map:     /rtabmap/octomap_full")
        self.get_logger().info("  2D map:     /rtabmap/octomap_grid (Z-filterable)")
        self.get_logger().info("  LiDAR:      /scan (360° + directional, body-frame corrected)")
        self.get_logger().info("  Depth:      /depth_camera (reactive avoidance)")
        self.get_logger().info("  Tracking:   odom-timeout based (1.5s threshold)")

        # ============================================
        # Continuous Offboard Heartbeat Timer
        # ============================================
        self._offboard_active = False
        self._current_setpoint = [0.0, 0.0, -1.5, 0.0]
        self._heartbeat_timer = self.create_timer(0.1, self._heartbeat_timer_cb)

    def _heartbeat_timer_cb(self):
        """10Hz heartbeat timer — keeps offboard mode alive continuously."""
        if not self._offboard_active:
            return
        self.publish_offboard_heartbeat()
        x, y, z, yaw = self._current_setpoint
        self.publish_setpoint(x, y, z, yaw=yaw)

    def start_offboard_stream(self, x=0.0, y=0.0, z=-1.5, yaw=0.0):
        """Start continuous offboard heartbeat + setpoint streaming."""
        self._current_setpoint = [float(x), float(y), float(z), float(yaw)]
        self._offboard_active = True
        self.get_logger().debug("Offboard stream started")

    def update_setpoint(self, x, y, z, yaw=0.0):
        """Update the continuously-streamed setpoint."""
        self._current_setpoint = [float(x), float(y), float(z), float(yaw)]

    def stop_offboard_stream(self):
        """Stop the continuous offboard heartbeat (before landing/disarm)."""
        self._offboard_active = False
        self.get_logger().debug("Offboard stream stopped")

    # ============================================
    # SLAM Callbacks
    # ============================================
    def _odom_cb(self, msg):
        self.odom = msg
        # FIX 2: Record timestamp of last odometry for tracking detection
        self._last_odom_time = self.get_clock().now()
        # If we were tracking-lost and odom resumes, recover
        if self.tracking_lost:
            self.tracking_lost = False
            self.get_logger().info("RTAB-Map tracking RECOVERED (odometry resumed)")

    def _loc_pose_cb(self, msg):
        """Global localization pose — covariance DECREASES after loop closures."""
        with self._covariance_lock:
            self.global_pose_covariance = np.array(msg.pose.covariance, dtype=np.float64)
        # FIX 2: Also counts as tracking-alive signal
        self._last_loc_pose_time = self.get_clock().now()
        if self.tracking_lost:
            self.tracking_lost = False
            self.get_logger().info("RTAB-Map tracking RECOVERED (localization pose received)")

    def _map_cb(self, msg):
        """2D projected occupancy grid from OctoMap."""
        w, h = msg.info.width, msg.info.height
        raw = np.array(msg.data, dtype=np.int8).reshape((h, w))
        known = int(np.sum(raw >= 0))
        # FIX 3: Atomic update of all map-related fields
        with self._map_lock:
            self.grid_info = msg.info
            self.occupancy_grid = raw
            self.total_cells = w * h
            self.known_cells_count = known

    def _octomap_cb(self, msg):
        self.octomap_data = msg

    def _frontier_cb(self, msg):
        self.frontier_count = msg.width
        self.frontier_points = msg

    def _info_cb(self, msg):
        """SLAM info callback — track loop closures.

        FIX 2: Tracking-loss detection has been moved OUT of this callback.
        The old approach counted consecutive frames where loop_closure_id == 0,
        but that's the normal state (loop closures are rare).  It triggered
        false tracking-loss after ~1.5s of normal exploration.

        Tracking is now detected via odometry timeout in is_tracking_lost().
        """
        self.loop_closure_id = msg.loop_closure_id
        # FIX 2 (secondary): Use msg.proximity_detection_id, not self.
        # The old code checked self.proximity_detection_id which was the
        # stale value from the *previous* callback.
        self.proximity_detection_id = msg.proximity_detection_id

    # ============================================
    # Sensor Callbacks (LiDAR + Depth)
    # ============================================
    def _lidar_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        if np.any(valid):
            min_range = float(np.min(ranges[valid]))
        else:
            min_range = float('inf')

        # FIX 3: Atomic update of all LiDAR fields together
        with self._lidar_lock:
            self.lidar_angle_min = msg.angle_min
            self.lidar_angle_max = msg.angle_max
            self.lidar_angle_increment = msg.angle_increment
            self.lidar_range_min = msg.range_min
            self.lidar_range_max = msg.range_max
            self.lidar_ranges = ranges
            self.lidar_min_range = min_range

    def _depth_cb(self, msg):
        try:
            h, w = msg.height, msg.width
            if msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w))
            elif msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w)).astype(np.float32) / 1000.0
            else:
                return

            self.depth_image = depth
            cy, cx = h // 4, w // 4
            center = depth[cy:3*cy, cx:3*cx]
            valid = np.isfinite(center) & (center > 0.1) & (center < 20.0)
            if np.any(valid):
                self.depth_min_distance = float(np.min(center[valid]))
            else:
                self.depth_min_distance = float('inf')
        except Exception:
            pass

    # ============================================
    # PX4 Callbacks
    # ============================================
    def _local_pos_cb(self, msg):
        self.local_position = msg

    def _status_cb(self, msg):
        self.vehicle_status = msg

    # ============================================
    # PX4 Command Methods
    # ============================================
    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x, y, z, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def publish_command(self, command, **kwargs):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = kwargs.get("param1", 0.0)
        msg.param2 = kwargs.get("param2", 0.0)
        msg.param3 = kwargs.get("param3", 0.0)
        msg.param4 = kwargs.get("param4", 0.0)
        msg.param5 = kwargs.get("param5", 0.0)
        msg.param6 = kwargs.get("param6", 0.0)
        msg.param7 = kwargs.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def arm(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("ARM command sent")

    def disarm(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("DISARM command sent")

    def engage_offboard(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("OFFBOARD mode command sent")

    def land(self):
        self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("LAND command sent")

    # ============================================
    # Data Accessor Methods — Position & Covariance
    # ============================================
    def get_drone_position(self):
        if self.local_position is None:
            return None
        return np.array([
            self.local_position.x,
            self.local_position.y,
            self.local_position.z,
        ], dtype=np.float64)

    def get_drone_yaw(self):
        if self.local_position is None:
            return 0.0
        return float(self.local_position.heading)

    def get_covariance_trace(self):
        """Full 6-DOF covariance trace: position (x,y,z) + orientation (r,p,yaw).

        FIX 5: Was [0, 7, 14] (XYZ only).  Yaw uncertainty (index 35) is
        critical for SLAM — if the drone is unsure which direction it faces,
        the map gets locally distorted.  Including all 6 DOF gives a more
        accurate picture of overall localization quality.

        NOTE: This changes the magnitude of the trace.  COV_MAX_FOR_NORM
        in active_slam_env.py may need re-tuning (try 3.0 instead of 2.0).
        """
        diag_indices = [0, 7, 14, 21, 28, 35]
        with self._covariance_lock:
            return float(sum(self.global_pose_covariance[i] for i in diag_indices))

    def get_covariance_trace_normalized(self, max_val=1.0):
        raw = self.get_covariance_trace()
        return float(np.clip(raw, 0.0, max_val) / max_val)

    def is_armed(self):
        if self.vehicle_status is None:
            return False
        return self.vehicle_status.arming_state == 2

    def get_altitude(self):
        pos = self.get_drone_position()
        if pos is None:
            return None
        return float(-pos[2])

    def is_tracking_lost(self):
        """Returns True if RTAB-Map has stopped producing odometry.

        FIX 2 (rewritten):
        Old approach counted consecutive frames where loop_closure_id == 0
        and proximity_detection_id == 0 in _info_cb.  But both being 0 is
        the NORMAL state — loop closures are rare (maybe 1 per 20-50 frames).
        The counter hit 45 within ~1.5s and falsely declared tracking lost
        during normal exploration of new areas.

        New approach: check whether ANY odometry message has arrived within
        the last 1.5s.  When RTAB-Map truly loses tracking, it stops
        publishing /rtabmap/odom entirely — an unambiguous signal.
        """
        if self._last_odom_time is None:
            # Haven't received first odom yet — don't panic during startup
            return False

        now = self.get_clock().now()
        elapsed = (now - self._last_odom_time).nanoseconds / 1e9

        if elapsed > self._tracking_lost_timeout and not self.tracking_lost:
            self.tracking_lost = True
            self.get_logger().warn(
                f"RTAB-Map tracking LOST — no odometry for {elapsed:.1f}s")

        return self.tracking_lost

    # ============================================
    # Data Accessor Methods — Map (thread-safe)
    # ============================================
    def get_grid_snapshot(self):
        """Return a consistent snapshot of (grid, grid_info, known, total).

        FIX 3: Ensures the grid and its metadata are from the same callback
        invocation, preventing mismatched coordinate transforms.
        """
        with self._map_lock:
            return (self.occupancy_grid, self.grid_info,
                    self.known_cells_count, self.total_cells)

    # ============================================
    # Data Accessor Methods — LiDAR
    # ============================================
    def get_min_lidar_range(self):
        with self._lidar_lock:
            return self.lidar_min_range

    def get_lidar_range_in_direction(self, target_angle_rad, world_frame=True):
        """Get min LiDAR range within a ±15° cone around the target direction.

        FIX 1 (CRITICAL):
        LaserScan angles are in the LiDAR BODY frame (angle 0 = sensor forward,
        which rotates with the drone).  But every caller in the codebase passes
        world-frame yaw (e.g. π/2 = east in the world).

        Old code compared world yaw directly against body-frame scan angles,
        meaning the directional safety check looked at the WRONG sector
        entirely.  The drone could fly into a wall while the check reported
        "all clear" because it was examining a sideways/backward sector.

        Now: when world_frame=True (default), we subtract the drone's current
        yaw to convert to body frame before querying.  All existing callers
        pass world-frame angles so they work correctly without changes.

        Args:
            target_angle_rad: Direction to check (radians).
            world_frame: If True (default), target_angle_rad is in NED world
                         frame and will be converted to body frame internally.
                         If False, it's already in LiDAR body frame.
        """
        with self._lidar_lock:
            if self.lidar_ranges is None or self.lidar_angle_increment == 0.0:
                return float('inf')

            # FIX 1: Convert world frame → body frame
            if world_frame:
                drone_yaw = self.get_drone_yaw()
                target_angle_rad = target_angle_rad - drone_yaw

            cone_half = 0.2618  # ~15 degrees
            ranges = self.lidar_ranges
            n = len(ranges)

            min_range = float('inf')
            for i in range(n):
                angle = self.lidar_angle_min + i * self.lidar_angle_increment
                diff = (angle - target_angle_rad + np.pi) % (2 * np.pi) - np.pi
                if abs(diff) <= cone_half:
                    r = ranges[i]
                    if np.isfinite(r) and self.lidar_range_min <= r <= self.lidar_range_max:
                        min_range = min(min_range, r)

            return float(min_range)

    def get_lidar_ranges_array(self):
        with self._lidar_lock:
            if self.lidar_ranges is None:
                return None
            return self.lidar_ranges.copy()

    # ============================================
    # Data Accessor Methods — Depth Camera
    # ============================================
    def get_depth_min_distance(self):
        return self.depth_min_distance

    # ============================================
    # Z-Filter for Altitude Slicing
    # ============================================
    def set_z_filter(self, min_z, max_z):
        """Set RTAB-Map grid height filter. Waits for confirmation.

        FIX 4: Old code called call_async() and never awaited the result,
        then immediately set z_filter_active = True assuming success.
        Now we spin-wait for the result with a timeout.
        """
        if not self.set_params_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("RTAB-Map parameter service not available for Z-filter")
            return False

        params = [
            Parameter(
                name="Grid/MinGroundHeight",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=str(min_z))),
            Parameter(
                name="Grid/MaxObstacleHeight",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=str(max_z))),
        ]

        request = SetParameters.Request(parameters=params)
        future = self.set_params_client.call_async(request)

        # FIX 4: Wait for result instead of fire-and-forget
        timeout_sec = 2.0
        start = _time.time()
        while not future.done():
            elapsed = _time.time() - start
            if elapsed > timeout_sec:
                self.get_logger().warn(
                    f"Z-filter set_parameters timed out after {timeout_sec}s")
                return False
            _time.sleep(0.05)

        result = future.result()
        if result is None:
            self.get_logger().warn("Z-filter set_parameters returned None")
            return False

        # Only record success after confirmation
        self.z_filter_min = min_z
        self.z_filter_max = max_z
        self.z_filter_active = True

        self.get_logger().info(f"Z-filter set (confirmed): [{min_z:.2f}m, {max_z:.2f}m]")
        return True

    def get_z_filter_state(self):
        return self.z_filter_min, self.z_filter_max, self.z_filter_active

    # ============================================
    # Frontier Point Extraction (3D → 2D)
    # NOTE (FIX 6): This method is currently unused — the env and all
    # baselines perform their own BFS frontier detection on the occupancy
    # grid.  Kept because RTAB-Map's native 3D frontiers from
    # /rtabmap/octomap_global_frontier_space are potentially more accurate
    # than 2D BFS and could replace the custom detection in the future.
    # ============================================
    def get_frontier_points_2d(self, z_min=None, z_max=None):
        if self.frontier_points is None:
            return np.empty((0, 2), dtype=np.float32)

        msg = self.frontier_points
        if msg.width == 0:
            return np.empty((0, 2), dtype=np.float32)

        try:
            x_off = y_off = z_off = None
            for field in msg.fields:
                if field.name == 'x':
                    x_off = field.offset
                elif field.name == 'y':
                    y_off = field.offset
                elif field.name == 'z':
                    z_off = field.offset

            if x_off is None or y_off is None:
                return np.empty((0, 2), dtype=np.float32)

            point_step = msg.point_step
            data = msg.data
            n_points = msg.width * msg.height
            points = []

            for i in range(n_points):
                base = i * point_step
                x = struct.unpack_from('f', data, base + x_off)[0]
                y = struct.unpack_from('f', data, base + y_off)[0]

                if z_off is not None and z_min is not None and z_max is not None:
                    z = struct.unpack_from('f', data, base + z_off)[0]
                    if z < z_min or z > z_max:
                        continue

                points.append([x, y])

            if len(points) == 0:
                return np.empty((0, 2), dtype=np.float32)
            return np.array(points, dtype=np.float32)

        except Exception:
            return np.empty((0, 2), dtype=np.float32)