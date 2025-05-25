#!/usr/bin/env python3

#find the position of the aruco wrt camera   
# odometry transformation, base to camera TF, camera to aruco. listen directly from the camera frame to the world 
# aruco box height is 15cm 
import rospy
import numpy as np
import cv2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import tf
from config import *


class ArucoDetection:
    def __init__(self):
        

        #SUBSCRIBERS    /turtlebot/kobuki/realsense/color/image_color
        #subscribe to camera image for the aruco detection
        self.image_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_color", Image, self.imageToCV) 
    
        #subscribe to camera info to get the camera matrix and distortion coefficients
        self.camera_info_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/camera_info", CameraInfo, self.camerainfoCallback) 
    
        #subscribe to the odometry topic to use later for transformation from world NED to robot base_footprint
        self.odom_sub = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.odomCallback) 

        #PUBLISHERS 
        #publish pose of aruco marker in the world frame 
        self.marker_pub = rospy.Publisher("/aruco_pose", PoseStamped, queue_size=1) 
        
        #define aruco dictionary and parameters 
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL) 
        self.aruco_params = cv2.aruco.DetectorParameters() 
        
        #bridge object to convert the image from ros to cv2 format
        self.bridge = CvBridge() 
        
        #aruco detector object
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params) 
        
        #initialise the camera matrix and distortion coefficients for pose estimation + camera size 
        self.camera_matrix  = None 
        self.dist_coeffs    = None 
     
        #listen to tf directly from the camera frame to the world frame
        self.tf_listener = tf.TransformListener()
        print("Waiting for tf listener")
    
    #odometry transformation  
    def odomCallback(self, odom): 
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                            odom.pose.pose.orientation.y,
                                                            odom.pose.pose.orientation.z,
                                                            odom.pose.pose.orientation.w])
        self.eta = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw]).reshape(-1,1)
        self.robot_orientation = np.array([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
        self.robot_translation = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
    
    #transform the obtained image to cv2 format to be used by the aruco detector
    def imageToCV(self, image): 
        """
        converts image format from ros to cv2

        Args:
            image: The image to be saved
        """
        print("Image received")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough').copy()
            self.publishAruco(cv_image)
        except CvBridgeError as err:
            print(err)
            
    
    def camerainfoCallback(self, data): 
        """
        Callback from camera to get the K matrix and distortion coefficients

        Args:
            data: The camera info message.
        
        """
        self.camera_matrix  = np.array(data.K).reshape(3,3) #get the camera matrix from the camera info topic 
        self.dist_coeffs    = np.array(data.D) 
        self.camera_info_sub.unregister() 
        
    def publishAruco(self, cv_image): 
        """
        detect the aruco marker in the image obtained from the rgb camera
        Args:
            cv_image: converted image
        """
        
        corners, ids, _ = cv2.aruco.detectMarkers(cv_image,self.aruco_dict)    # detect the marker 
        
        if len(corners)>0: #if the aruco is detected
            # print("Aruco detected", ids)
            
            #estimate the aruco pose (returns the rotation and translation vector) 
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.camera_matrix, self.dist_coeffs)
            
            for i in range(len(ids)): #for each marker detected
                rvec_i = rvec[i,:,:].reshape(3,1)
                tvec_i = tvec[i,:,:].reshape(3,1)
                # Move to center of the box
                tvec_i[2] += 0.07 / 2.0

                object_pose_world = self.compute_object_pose_in_world(rvec_i, tvec_i)

                # Extract translation and rotation (quaternion) from the object's pose in the world frame
                object_translation_world = object_pose_world[:3, 3]
  
                object_quaternion_world = tf.transformations.quaternion_from_matrix(object_pose_world)

                #################### POSE PUBLISHER OF ARUCO IN WORLD NED FRAME #############################
                aruco_position = PoseStamped()
                aruco_position.header.frame_id = "world_ned"        #is the frame_id correct?
                aruco_position.pose.position.x = object_translation_world[0]
                aruco_position.pose.position.y = object_translation_world[1]
                aruco_position.pose.position.z = object_translation_world[2]
                aruco_position.pose.orientation.x = object_quaternion_world[0]
                aruco_position.pose.orientation.y = object_quaternion_world[1]
                aruco_position.pose.orientation.z = object_quaternion_world[2]
                aruco_position.pose.orientation.w = object_quaternion_world[3]
                self.marker_pub.publish(aruco_position)
            
            print ("Aruco ID:", ids)
            
        else :
            print("No Aruco detected")
            
    # Step 3: Compute the object's position in the world frame
    def compute_object_pose_in_world(self, rvec, tvec):
        # Get transformation matrix object in camera frame
        rotation_matrix_object_in_camera = rvec_to_rot_matrix(rvec)
        object_in_camera = create_homogeneous_transform(rotation_matrix_object_in_camera, tvec)

        # Get transformation matrix camera in base footprint frame
        camera_rot_matrix = tf.transformations.quaternion_matrix(np.array([0.500, 0.500, 0.500, 0.500]))
        camera_in_robot = np.eye(4)
        camera_in_robot[:3, :3] = camera_rot_matrix[:3, :3]
        camera_in_robot[0:3, 3] = np.array([0.136, -0.033, -0.116])

        # Get transformation matric robot base footprint in the map frame
        # Convert quaternion to rotation matrix
        robot_rot_matrix = tf.transformations.quaternion_matrix(self.robot_orientation)
        # Create homogeneous transformation matrix for robot in the world frame
        robot_in_world = np.eye(4)
        robot_in_world[:3, :3] = robot_rot_matrix[:3, :3]
        robot_in_world[0:3, 3] = self.robot_translation

        # Compute the object's pose in the world frame
        object_in_robot = np.dot(camera_in_robot, object_in_camera)
        object_in_world = np.dot(robot_in_world, object_in_robot)
        return object_in_world


# Step 1: Convert rvec to a rotation matrix
def rvec_to_rot_matrix(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return rotation_matrix

# Step 2: Construct the homogeneous transformation matrix
def create_homogeneous_transform(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector.T
    return transformation_matrix
    
if __name__ == "__main__":

    print("ARUCO POSE DETECTOR NODE INITIALIZED")
    # print(cv2.__version__)
    # print(cv2.__file__)  
    rospy.init_node('aruco_pose_detector_node')   
    
    node = ArucoDetection()

    rospy.spin()                              
                              