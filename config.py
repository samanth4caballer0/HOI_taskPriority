import numpy as np

limit_joint1_upper =  1.571
limit_joint1_lower = -1.571
limit_joint2_upper =  0.050
limit_joint2_lower = -1.571
limit_joint3_upper =  0.050
limit_joint3_lower = -1.571
limit_joint4_upper =  1.571
limit_joint4_lower = -1.571

limit_joint_threshold_activate      = 0.05
limit_joint_threshold_deactivate    = 0.08

# Task hierarchy definition
limit_range_joint1   = np.array([limit_joint1_lower, limit_joint1_upper]).reshape(1,2)
limit_range_joint2   = np.array([limit_joint2_lower, limit_joint2_upper]).reshape(1,2)
limit_range_joint3   = np.array([limit_joint3_lower, limit_joint3_upper]).reshape(1,2)
limit_range_joint4   = np.array([limit_joint4_lower, limit_joint4_upper]).reshape(1,2)

threshold_joint      = np.array([limit_joint_threshold_activate, limit_joint_threshold_deactivate]).reshape(2,1)

BASE_CONFIG_WEIGHT_BASE_ROTATE      = 10.000
BASE_CONFIG_WEIGHT_BASE_TRANSLATE   = 50.000
BASE_CONFIG_WEIGHT_JOINT_1           = 0.500
BASE_CONFIG_WEIGHT_JOINT_2           = 1.000
BASE_CONFIG_WEIGHT_JOINT_3           = 1.000
BASE_CONFIG_WEIGHT_JOINT_4           = 1.000

EE_POS_WEIGHT_BASE_ROTATE      = 10.0
EE_POS_WEIGHT_BASE_TRANSLATE   = 150.0
EE_POS_WEIGHT_JOINT_1          = 0.5
EE_POS_WEIGHT_JOINT_2           = 1.0
EE_POS_WEIGHT_JOINT_3           = 1.0
EE_POS_WEIGHT_JOINT_4           = 1.0

# joint_state_topic  = "/turtlebot/joint_states"
# aruco_pose_topic   = "/aruco_pose"
# task_topic         = "None"
# cmd_vel_topic      = "/cmd_vel"
# cmd_dq_topic       = "/turtlebot/swiftpro/joint_velocity_controller/command"

FRAME_MAP               = "map"
FRAME_BASE_FOOTPRINT    = "turtlebot/kobuki/base_footprint"
FRAME_EE                = "EE"

color_camera_SIL_topic  = "/turtlebot/kobuki/realsense/color/image_color"
color_camera_HIL_topic  = "/turtlebot/kobuki/realsense/color/image_raw"
camera_info_topic       = "/turtlebot/kobuki/realsense/color/camera_info"
odom_SIL_topic          = "/odom"
odom_HIL_topic          = "/state_estimation"
aruco_pose_topic        = "/aruco_pose"
task_topic              = "/task"
rviz_goal_topic         = "/move_base_simple/goal"

suction_service         = '/turtlebot/swiftpro/vacuum_gripper/set_pump'

joint_state_topic       = "/turtlebot/joint_states"
cmd_vel_topic           = "/cmd_vel"
cmd_dq_topic            = "/turtlebot/swiftpro/joint_velocity_controller/command"
EEposition_marker_topic = '~EEposition_point_marker'
point_marker_topic      = '~desierd_point_marker'
task_error_topic        = "/error_topic"

MODE                    = "SIL"
# Aruco
MAKER_SIZE              = 0.05
BOX_WIDTH               = 0.07
BOX_LENGTH              = 0.07
BOX_HEIGHT              = 0.07

CAM_BASE_X              = 0.136
CAM_BASE_Y              = -0.033
CAM_BASE_Z              = -0.116

CAM_BASE_QX             = 0.500
CAM_BASE_QY             = 0.500
CAM_BASE_QZ             = 0.500
CAM_BASE_QW             = 0.500


# MOBILE MANIPULATOR
MANI_BX                 = 0.0132        # [met]
MANI_BZ                 = 0.1080        # [met]
MANI_D1                 = 0.1420        # [met]
MANI_D2                 = 0.1588        # [met]
MANI_MZ                 = 0.0722        # [met]
MANI_MX                 = 0.0565        # [met]
MANI_ALPHA_SIL          = -np.pi/2.0    # [rad]
MANI_ALPHA_HIL          = np.pi/2.0     # [rad]


# BEHAVIOR TREE
SCAN_INIT_HEADING       = -np.pi/2.0 
SCAN_HEADING_STEP       = np.pi/2.0
SCAN_HEADING_RANGE_LOWER = -0.2         # [rad]
SCAN_HEADING_RANGE_UPER = 0.0           # [rad]
SCAN_HEADING_ERROR      = 0.2           # [rad]

MANI_SAFE_HEIGHT        = -0.35         # [met]
MANI_PICK_HEIGHT        = -0.13         # [met]

# BASE ORIENTATION CONTROLLER CONFIG
BASE_ORI_GAIN           = 0.1
BASE_ORI_FEEDFORWARD    = 0.0

# BASE CONFIGURATION CONTROLLER CONFIG
BASE_CONFIG_GAIN_X                  = 0.1
BASE_CONFIG_GAIN_Y                  = 0.1
BASE_CONFIG_GAIN_HEADING            = 0.4
BASE_CONFIG_FEEDFORWARD_X           = 0.0
BASE_CONFIG_FEEDFORWARD_Y           = 0.0
BASE_CONFIG_FEEDFORWARD_HEADING     = 0.0
BASE_CONFIG_DIS_ERROR_FINISH        = 0.4
BASE_CONFIG_HEADING_ERROR_FINISH    = 0.3

# EE POSITION CONTROLLER CONFIG
EE_POS_GAIN_X                       = 0.2
EE_POS_GAIN_Y                       = 0.2
EE_POS_GAIN_Z                       = 0.1
EE_POS_FEEDFORWARD_X                = 0.0
EE_POS_FEEDFORWARD_Y                = 0.0
EE_POS_FEEDFORWARD_Z                = 0.0
EE_POS_ERROR_FINISH                 = 0.1
EE_POS_ERROR_PICK_OBJ               = 0.02

# GOAL PLACE
GOAL_PLACE_X                        = 3.0
GOAL_PLACE_Y                        = 1.0

# TASK
N_OBJECT                            = 1.0
