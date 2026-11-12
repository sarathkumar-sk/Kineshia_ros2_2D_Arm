# Periodic status of the arm controller (published on /controller_status).

std_msgs/Header header

# Lifecycle/motion state: "idle", "moving", "holding", "error"
string state
# Active command interface: "position", "velocity" or "current"
string control_mode
# Active hardware backend: "sim" or "gazebo"
string backend

# Last requested Cartesian target and the target actually used after the IK
# projected it onto the reachable workspace.
geometry_msgs/Point requested_target
geometry_msgs/Point reachable_target
bool target_projected

# Trajectory progress in [0, 1] for the current motion.
float64 progress

# Per-joint tracking error q_ref - q_measured (rad).
float64[] tracking_error

# Virtual gripper / payload
bool gripper_closed
bool object_attached
geometry_msgs/Point object_position

# Control-loop timing (measured, ms) over a rolling window.
float64 control_period_target_ms
float64 control_period_mean_ms
float64 control_period_std_ms
float64 control_period_max_ms
float64 control_jitter_p99_ms
float64 control_loop_exec_mean_ms

# Human-readable message about the last event (e.g. why a target was rejected).
string message
