import tf2_ros
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped, Twist
from sensor_msgs.msg import JointState
from math import sin, cos
import tf
import numpy as np
import sympy as sp 
from common import *


class TP_controller:
    def __init__(self):
        rospy.init_node("TP_control_node")
        rospy.loginfo("Starting Task Priority Controller....")
        
        #self.publish_static_transform()  
        self.MM = MobileManipulator() 
        #self.ee_pub = rospy.Publisher('/end_effector', Odometry, queue_size=10)
        self.ee_pub = rospy.Publisher('/end_effector', PoseStamped, queue_size=10)
        self.joint_velocity_pub = rospy.Publisher('/turtlebot/swiftpro/joint_velocity_controller/command', Float64MultiArray, queue_size=10)
        # Timer for TP controller (Velocity Commands)
        rospy.Subscriber('/desired_sigma',PoseStamped , self.set_desired_sigma)
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_pos_callback)
        
        odom_sim_topic = "/turtlebot/kobuki/odom" # odometry topic for simulation
        #subscribe to the odometry topic to use later for transformation from world NED to robot base_footprint
        self.odom_sub = rospy.Subscriber(odom_sim_topic, Odometry, self.odomCallback) 

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

        ee_pose.header.frame_id = 'world_ned'
        ee_pose.header.stamp = rospy.Time.now()        

        ee_pose.pose.position = Point(x, y, z)
        # Convert yaw to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        ee_pose.pose.orientation = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        
        #publish the odom message 
        self.ee_pub.publish(ee_pose)

        # control part
        # Publish the joint velocity command
        #dq  = np.zeros((self.MM.dof, 1))

        """ J = i.getJacobian()           # task full Jacobian
        Jbar = (J @ null_space)                      # projection of task in null-space
        Jbar_inv = DLS(Jbar, 0.1)                    # pseudo-inverse or DLS
        dq += Jbar_inv @ ((i.getK()@i.getError()-J@dq) + i.ff)      # calculate quasi-velocities with null-space tasks execution """
        
        print ("total jacobian: ", self.MM.get_MMJacobian())
        
        if self.MM.d_sigma is not None:
            J = self.MM.get_armJacobian()

            Jinv = self.weighted_DLS(J,0.004) 
            
            #print ("Jinv: ", Jinv)
            #print ("error: ", self.MM.get_error())

            # dq = np.linalg.pinv(J) @ desired_vel
            dq = Jinv @ self.MM.get_error()

            
            # publish the joint velocity command
            joint_velocity = Float64MultiArray()
            joint_velocity.data = dq.flatten().tolist()

            self.joint_velocity_pub.publish(joint_velocity)
    

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

    #odometry transformation  
    def odomCallback(self, odom): 
        
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                    odom.pose.pose.orientation.y,
                                                    odom.pose.pose.orientation.z,
                                                    odom.pose.pose.orientation.w])
        self.MM.eta = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z, yaw]).reshape(-1,1)
        
   
        
    def joint_pos_callback(self, msg):
        #filter out the passive joints 
        if msg.name[0] == 'turtlebot/swiftpro/joint1':

        #extract the active joint position array 
            joint_pos = np.array([msg.position[0], msg.position[1], msg.position[2], msg.position[3]])
            
        #store it as a variable 
            self.MM.q = joint_pos.reshape(4,1)
    
    def set_desired_sigma(self, msg):  
        # Extract the desired end-effector pose from the message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        yaw = tf.transformations.euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])[2]
        
        # Set the desired end-effector pose in the MobileManipulator object
        self.MM.d_sigma = np.zeros((4))

        self.MM.d_sigma[0] = x
        self.MM.d_sigma[1] = y
        self.MM.d_sigma[2] = z
        self.MM.d_sigma[3] = yaw
       
    def weighted_DLS(self,A, damping):
        '''
            Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

            Arguments:
            A (Numpy array): matrix to be inverted
            damping (double): damping factor

            Returns:
            (Numpy array): inversion of the input matrix
        '''
        return  A.T @ np.linalg.inv(A  @ A.T + damping**2 * np.eye(np.shape(A)[0], np.shape(A)[0]))


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

        # Vector of desired end-effector pose (position & orientation)
        self.d_sigma        = None


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
        self.bmz = 0.198  # 
        self.bmx = 0.0507 # distance from the robot base frame to arm base frame
        
        # end effector offsets
        self.mx = 0.0565  
        self.mz = 0.0722

        # distance of the links
        d1 = 0.1420
        d2 = 0.1588
        
        # Calculate the lengths of the manipulator arm segments
        l1_x = d1 * sin(-q2)    #projection of d1 on x-axis
        l2_x = d2 * cos(q3)     #projection of d2 on x-axis
        self.l = (self.bx + (l1_x) + (l2_x) + self.mx) #diagonal between the base and the end effector

        # #forward kinematics to get ee position
        # self.x = self.l * cos(q1)
        # self.y = self.l * sin(q1) 
        # self.z = (-self.bz - (d1 * cos(-q2)) - (d2 * sin(q3)) + self.mz - self.bmz) 
        # self.yaw = q1 + q4 + alpha
         
        
        ###############include full MM jacobuan
        eta_x = self.eta[0,0]
        eta_y = self.eta[1,0]
        eta_z = self.eta[2,0]
        theta = self.eta[3,0]
        alpha = -deg90

        self.x = self.l * cos(q1 + theta + alpha) + eta_x + self.bmx * cos(theta)
        self.y = self.l * sin(q1 + theta + alpha) + eta_y + self.bmx * sin(theta)
        self.z = (-self.bz - (d1 * cos(-q2)) - (d2 * sin(q3)) + self.mz - self.bmz)  # plus or minus mz???????????
        self.yaw = q1 + q4 + alpha + theta
        #this returns only the derivation by the q's, not the mobile base joints 
        
        return self.x, self.y, self.z, self.yaw
    
    class MobileBaseParams:
        def __init__(self) -> None:
            self.bmx = 0.0507       # [met]
            self.bmz = -0.198       # [met]

            self.theta    = np.array([   deg90,        0])
            self.d        = np.array([self.bmz, self.bmx])
            self.a        = np.array([       0,        0])
            self.alpha    = np.array([   deg90,   -deg90])

            self.revolute   = [True, False]
            self.dof        = len(self.revolute)
            
    
    def get_armJacobian(self):
        # deifne symbolic variables
        q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')
        
        
        # define symbolic transformation matrices  
        q = sp.Matrix([q1, q2, q3, q4])
        
        # define links lenghts (arm constants)
        l1 = 0.1420
        l2 = 0.1588
        
        self.bx = 0.0132
        self.bz = 0.108

        #
        self.bmz = 0 # 0.198        # z-distance offset of the arm base frame from the robot base frame
        self.bmx = 0 # 0.0507       # x-distance from the robot base frame to arm base frame
        
        # end effector offsets
        self.mx = 0.0565  
        self.mz = 0.0722

        # Calculate the lengths of the manipulator arm segments
        l1_x = l1 * sp.sin(-q2)    #projection of d1 on x-axis
        l2_x = l2 * sp.cos(q3)     #projection of d2 on x-axis
        l = (self.bx + (l1_x) + (l2_x) + self.mx) #diagonal between the base and the end effector
        
        
        #forward kinematics to get ee position
        x = l * sp.cos(q1)
        y = l * sp.sin(q1) 
        z = (-self.bz - (l1 * sp.cos(-q2)) - (l2 * sp.sin(q3)) + self.mz - self.bmz) 
        yaw = q1 + q4 
        
        # task vector 
        task_vector = sp.Matrix([x, y, z, yaw])
        # Jacobian matrix
        J_eq = task_vector.jacobian(q)
        
        q_values = {
            q1: float(self.q[0,0]),
            q2: float(self.q[1,0]),
            q3: float(self.q[2,0]),
            q4: float(self.q[3,0])
        }
        
        
        # debugging jacobian 
        q_values_manual = {
            q1: 0.0,
            q2: 0.0,
            q3: 0.0,
            q4: 0.0
        }
        
        J_num_manual = J_eq.subs(q_values_manual)
        #print("Jacobian at q = 0: ", J_num_manual)
        
        
        
        
        # Substitute numerical values
        J_num = J_eq.subs(q_values)

        # Convert to NumPy array for numerical operations
        J = np.array(J_num).astype(np.float64)
        return J
    
    def get_MMJacobian(self):
        # deifne symbolic variables
        q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')
        
        
        # define symbolic transformation matrices  
        q = sp.Matrix([q1, q2, q3, q4])
        
        # define links lenghts (arm constants)
        l1 = 0.1420
        l2 = 0.1588
        
        self.bx = 0.0132
        self.bz = 0.108

        #
        self.bmz =  0.198        # z-distance offset of the arm base frame from the robot base frame
        self.bmx =  0.0507       # x-distance from the robot base frame to arm base frame
        
        # end effector offsets
        self.mx = 0.0565  
        self.mz = 0.0722

        # Calculate the lengths of the manipulator arm segments
        l1_x = l1 * sp.sin(-q2)    #projection of d1 on x-axis
        l2_x = l2 * sp.cos(q3)     #projection of d2 on x-axis
        l = (self.bx + (l1_x) + (l2_x) + self.mx) #diagonal between the base and the end effector
        
        
        #new equations for full MM jacobian 
        eta_x = self.eta[0,0]
        eta_y = self.eta[1,0]
        eta_z = self.eta[2,0]
        theta = self.eta[3,0]
        alpha = -deg90

        x = l * sp.cos(q1 + theta + alpha) + eta_x + self.bmx * sp.cos(theta)
        y = l * sp.sin(q1 + theta + alpha) + eta_y + self.bmx * sp.sin(theta)
        z = (-self.bz - (l1 * sp.cos(-q2)) - (l2 * sp.sin(q3)) + self.mz - self.bmz) + eta_z # plus or minus mz???????????
        yaw = q1 + q4 + alpha + theta
        #this returns only the derivation by the q's, not the mobile base joints 
        
        
        # task vector 
        task_vector = sp.Matrix([x, y, z, yaw])
        # Jacobian matrix
        J_eq = task_vector.jacobian(q)
        
        q_values = {
            q1: float(self.q[0,0]),
            q2: float(self.q[1,0]),
            q3: float(self.q[2,0]),
            q4: float(self.q[3,0])
        }
        
        
        # debugging jacobian 
        q_values_manual = {
            q1: 0.0,
            q2: 0.0,
            q3: 0.0,
            q4: 0.0
        }
        
        J_num_manual = J_eq.subs(q_values_manual)
        #print("Jacobian at q = 0: ", J_num_manual)
        
        
        # Substitute numerical values
        J_num = J_eq.subs(q_values)

        # Convert to NumPy array for numerical operations
        J_arm = np.array(J_num).astype(np.float64)

        # Whole Jacobian 
        JB = self.getMbaseJacobian()
        print ("JB: ", JB)
        J = np.zeros((4, 6))
        J[:, 0] = JB[[0,1,2,-1],0].reshape((4))                        # derivertive by m1
        J[:, 1] = JB[[0,1,2,-1],1].reshape((4))                        # derivertive by m2

        J[0, 2:] = J_arm[0, :]
        J[1, 2:] = J_arm[1, :]
        J[2, 2:] = J_arm[2, :]
        J[3, 2:] = J_arm[3, :]
        
        return J
    
    def getMbaseJacobian(self):
        self.revolute   = [True, False]
        # Base kinematics
        x = float(self.eta[0])
        y = float(self.eta[1])
        yaw = float(self.eta[3])
        Tb = translation2D(x, y) @ rotation2D(yaw)

        # Modify the theta of the base joint, to account for an additional Z rotation
        theta = float(self.q[0,0]-deg90)

        # Combined system kinematics (DH parameters extended with base DOF)
        thetaExt    = np.concatenate([np.array([deg90,                                 0]), np.array([theta])])
        dExt        = np.concatenate([np.array([self.bmz, self.bmx]),    np.array([0])])
        aExt        = np.concatenate([np.array([0,                                     0]),    np.array([self.l])])
        alphaExt    = np.concatenate([np.array([deg90,                            -deg90]),    np.array([0])])

        self.T      = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)

        T  = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)
        JB = jacobian(T, self.revolute + [True])
        
        return JB
        
    
    def get_error(self):
        # Calculate the end effector pose
        x, y, z, yaw = self.x, self.y, self.z, self.yaw
        
        # Calculate the error between the desired and current end-effector pose

        error = np.array([[self.d_sigma[0] - x], [self.d_sigma[1] - y], [self.d_sigma[2] - z], [0]])
        print ("error: ", error)
        return error
            
    
if __name__ == "__main__":

    ros_node = TP_controller()      
    rospy.spin()