import tf2_ros
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from sensor_msgs.msg import JointState
from math import sin, cos
import tf
import numpy as np

class TP_controller:
    def __init__(self):
        rospy.init_node("TP_control_node")
        rospy.loginfo("Starting Task Priority Controller....")
        
        #self.publish_static_transform()  
        self.MM = MobileManipulator() 
        #self.ee_pub = rospy.Publisher('/end_effector', Odometry, queue_size=10)
        self.ee_pub = rospy.Publisher('/end_effector', PoseStamped, queue_size=10)
        # Timer for TP controller (Velocity Commands)
        rospy.Subscriber('/swiftpro/joint_states', JointState, self.joint_pos_callback)
        rospy.Timer(rospy.Duration(0.1), self.visualize)
        

    def visualize(self, _):
        
        # Calculate the end effector pose
        x, y, z, yaw = self.MM.getEndEffectorPose()
        
        # create odom message Using Odometry message
        """ ee_pose = Odometry()
        ee_pose.header.frame_id = 'swiftpro/manipulator_base_link'
        ee_pose.child_frame_id = 'ee_frame'
        ee_pose.pose.pose.position = Point(x, y, z)
        
        # Convert yaw to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        ee_pose.pose.pose.orientation = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]) """

        # create odom message Using Pose message
        ee_pose = PoseStamped()

        ee_pose.header.frame_id = 'swiftpro/manipulator_base_link'
        ee_pose.header.stamp = rospy.Time.now()        

        ee_pose.pose.position = Point(x, y, z)
        # Convert yaw to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        ee_pose.pose.orientation = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        
        #publish the odom message 
        self.ee_pub.publish(ee_pose)
    

    def publish_static_transform(self):
                
        static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transform = TransformStamped()
        
        # Set header information
        static_transform.header.stamp = rospy.Time.now()
        static_transform.header.frame_id = "swiftpro/manipulator_base_link"
        static_transform.child_frame_id = "footprint_new"
        
        # Set translation (no translation in this case)
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        
        # Set rotation (180° around X-axis)
        static_transform.transform.rotation.x = 0.0  # sin(π/2) = 1.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0  # sin(π/4) = 0.7071
        static_transform.transform.rotation.w = 1  # sin(π/4) = 0.7071
        
        # Broadcast the transform
        static_broadcaster.sendTransform(static_transform)
        rospy.loginfo("Published static transform from base_footprint to footprint_new")

        
        
    def joint_pos_callback(self, msg):
        #filter out the passive joints 
        if msg.name[0] == 'swiftpro/joint1':

        #extract the active joint position array 
            joint_pos = np.array([msg.position[0], msg.position[1], msg.position[2], msg.position[3]])
            
        #store it as a variable 
            self.MM.q = joint_pos.reshape(4,1)
            


class MobileManipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self):

        self.dof            = 6

        # Vector of joint positions (manipulator)
        self.q              = np.zeros((4, 1))

        # Vector of base pose (position & orientation)
        self.eta            = np.zeros((4, 1))


    def getEndEffectorPose(self):
        
        # Extract the joint angles
        q1 = self.q[0]
        q2 = self.q[1]
        q3 = self.q[2]
        q4 = self.q[3]
        alpha = 0 # -np.pi/2.0      #rotation of the end effector frame with respect to the base frame
        
        # offset from joint 1 to the first articulation 
        self.bx = 0.0132
        self.bz = 0.108

        #
        self.bmz = 0 # 0.198  # 
        self.bmx = 0 # 0.0507 # distance from the robot base frame to arm base frame
        
        # end effector offsets
        self.mx = 0.0565  
        self.mz = 0.0722

        # distance of the links
        d1 = 0.1420
        d2 = 0.1588
        
        # Calculate the lengths of the manipulator arm segments
        l1_x = d1 * sin(-q2)    #projection of d1 on x-axis
        l2_x = d2 * cos(q3)     #projection of d2 on x-axis
        l = (self.bx + (l1_x) + (l2_x) + self.mx) #diagonal between the base and the end effector
        
        
        #forward kinematics to get ee position
        x = l * cos(q1)
        y = l * sin(q1) 
        z = (-self.bz - (d1 * cos(-q2)) - (d2 * sin(q3)) + self.mz - self.bmz) 
        yaw = q1 + q4 + alpha
        
        return x, y, z, yaw
    
        
    
if __name__ == "__main__":

    ros_node = TP_controller()      
    rospy.spin()