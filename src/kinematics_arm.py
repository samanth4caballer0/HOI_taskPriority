import tf2_ros
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped, Twist
from sensor_msgs.msg import JointState
from math import sin, cos
import tf
import numpy as np
import sympy as sp 
from common import *
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tasks import Position3D, Orientation2D, JointLimit2D


class TP_controller:
    def __init__(self):
        rospy.init_node("TP_control_node")
        rospy.loginfo("Starting Task Priority Controller....")
        
        self.color_red = ColorRGBA()
        self.color_red.r = 1
        self.color_red.g = 0
        self.color_red.b = 0
        self.color_red.a = 1
        self.color_blue = ColorRGBA()
        self.color_blue.r = 0
        self.color_blue.g = 0
        self.color_blue.b = 1
        self.color_blue.a = 1 
        self.color = self.color_blue
                
        self.m = Marker()
        self.start_time = rospy.Time.now().to_sec()

        self.request_goal_pub = rospy.Publisher('/goal_request', Bool, queue_size=10)

        # Task definition
        # Joint limits definition for the joints limits tasks
        j1_limits = np.array([np.pi, -np.pi])
        j2_limits = np.array([np.pi, -np.pi])
        j3_limits = np.array([np.pi, -np.pi])
        j4_limits = np.array([np.pi, -np.pi])
        self.j1_limit = JointLimit2D("joint 1 limit", 0, j1_limits, tresholds=[0.03, 0.035])
        self.j2_limit = JointLimit2D("joint 2 limit", 1, j2_limits, tresholds=[0.03, 0.035])
        self.j3_limit = JointLimit2D("joint 3 limit", 2, j3_limits, tresholds=[0.03, 0.035])
        self.j4_limit = JointLimit2D("joint 4 limit", 3, j4_limits, tresholds=[0.03, 0.035])

        self.position_task = Position3D("cartesion 3D position",self.MM, 4)
        self.orientation_task = Orientation2D("orientation",self.MM, 4)
        self.tasks = [self.j1_limit, self.j2_limit, self.j3_limit, self.j4_limit,
                      self.position_task, self.orientation_task]

        sigma_pose = self.position_task.getDesired()
        self.MM.d_sigma[0] = sigma_pose[0]
        self.MM.d_sigma[1] = sigma_pose[1]
        self.MM.d_sigma[2] = sigma_pose[2]

        
        
        #self.publish_static_transform()  
        self.MM = MobileManipulator() 
        
        #self.ee_pub = rospy.Publisher('/end_effector', Odometry, queue_size=10)
        self.ee_pub = rospy.Publisher('/end_effector', PoseStamped, queue_size=10)
        self.joint_velocity_pub = rospy.Publisher('/turtlebot/swiftpro/joint_velocity_controller/command', Float64MultiArray, queue_size=10)
        self.path_pub = rospy.Publisher('/path', Marker, queue_size=10)
        self.request_goal_pub = rospy.Publisher('/goal_request', Bool, queue_size=10)

        
        # Timer for TP controller (Velocity Commands)
        rospy.Subscriber('/desired_sigma',PoseStamped , self.set_desired_sigma)
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_pos_callback)
        
        rospy.sleep(0.3) #wait for the joint state to be published
        self.request_goal_pub.publish(True) # request a new goal
        rospy.Timer(rospy.Duration(0.1), self.visualize) # Timer for visualization
        
        #command veloctiy publisher 
        self.cmd_vel= rospy.Publisher('/turtlebot/kobuki/commands/velocity', Twist, queue_size=10)
        
        odom_sim_topic = "/turtlebot/kobuki/odom" # odometry topic for simulation
        #subscribe to the odometry topic to use later for transformation from world NED to robot base_footprint
        self.odom_sub = rospy.Subscriber(odom_sim_topic, Odometry, self.odomCallback) 

        rospy.Timer(rospy.Duration(0.1), self.visualize)
        

    def visualize(self, _):
        
        for i in self.tasks[0:4]:
            if i.isActive():
                self.color = self.color_red
            else:
                self.color = self.color_blue
        
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

        position = Point(x, y, z)
        self.m.points.append(position)
        self.m.colors.append(self.color)  
        self.publish_path()
        
        ee_pose.pose.position = position
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
        
        # if self.MM.d_sigma is not None:
            
        #     for i in range(len(self.tasks)):
        #         # get the task
        #         task = self.tasks[i]
                
                
        #         # get the task jacobian
        #         J = task.get_jacobian()

        #         # get the null space jacobian
        #         null_space = task.get_null_space_jacobian()

        #         # get the joint velocity command
        #         dq = np.zeros((self.MM.dof, 1))
            
        #     # J = self.MM.get_armJacobian()
        #     J = self.MM.get_MMJacobian()
        #     Jinv = self.weighted_DLS(J,0.004) 
            
        #     #print ("Jinv: ", Jinv)
        #     #print ("error: ", self.MM.get_error())

        #     # dq = np.linalg.pinv(J) @ desired_vel
        #     # K = 0.2 
        #     # dq = Jinv @ (K * self.MM.get_error())         #6x1  
        #     dq = Jinv @ self.MM.get_error()                 #6x1
            
            
            
        #     #access the velocity of the base joints M1 (angular) M2 (linear in x direction) 
        #     M1 = dq[0,0] #angular velocity of the base joint
        #     M2 = dq[1,0]
            
        #     # publish the cmd_vel message
        #     cmd_vel = Twist()
        #     cmd_vel.linear.x = M2
        #     cmd_vel.angular.z = M1
        #     self.cmd_vel.publish(cmd_vel)
        #     print ("dq: ", dq)
            
        #     # publish the joint velocity command
        #     joint_velocity = Float64MultiArray()
        #     joint_velocity.data = dq[2:,0].flatten().tolist()

        #     self.joint_velocity_pub.publish(joint_velocity)
            
        #     if self.MM.error_magnitude < self.MM.threshold_distance:
        #         self.request_goal_pub.publish(True)
        #         self.m = Marker() 
        
        null_space = np.eye(self.MM.dof)                  # initial null space P (projector)
        dq = np.zeros(self.MM.dof).reshape(-1, 1)         # initial quasi-velocities

        for i in self.tasks:
            i.update(self.MM)                             # update task Jacobian and error
            if i.isActive():
                """ print ("i.getJacobian(): ", i.J)
                print ("null_space: ", null_space) """
                J = i.getJacobian()                          # task full Jacobian
                Jbar = (J @ null_space)                      # projection of task in null-space
                Jbar_inv = self.weighted_DLS(Jbar, 0.04)                    # pseudo-inverse or DLS
                """ print ("Jbar_inv: ", Jbar_inv)
                print ("j@dq: ", J@dq)
                print ("i.getError(): ", i.getError())
                print ("k: ", i.getK())""" 
                self.MM.get_error()
                dq += Jbar_inv @ ((i.getK()@i.getError()-J@dq) + i.ff)      # calculate quasi-velocities with null-space tasks execution
                null_space = null_space - np.linalg.pinv(Jbar) @ Jbar   # update null-space projector

        # publish the joint velocity command
        joint_velocity = Float64MultiArray()
        joint_velocity.data = dq.flatten().tolist()

        self.joint_velocity_pub.publish(joint_velocity)


        if self.MM.error_magnitude < self.MM.threshold_distance :
            print ("goal reached")
            self.start_time = rospy.Time.now().to_sec()
            self.request_goal_pub.publish(True)
            self.m = Marker() 

        # if 10 seconds have passed, ask for a new goal
        if rospy.Time.now().to_sec() - self.start_time > 10:
            self.request_goal_pub.publish(True)
            self.start_time = rospy.Time.now().to_sec()
            self.m = Marker()

    def publish_path(self):

        #self.m = Marker()
        self.m.header.frame_id = 'swiftpro/manipulator_base_link'
        self.m.header.stamp = rospy.Time.now()
        self.m.id = 0
        self.m.type = Marker.LINE_STRIP
        self.m.ns = 'path'
        self.m.action = Marker.DELETE
        self.m.lifetime = rospy.Duration(0)

        self.m.action = Marker.ADD
        self.m.scale.x = 0.02
        self.m.scale.y = 0.02
        self.m.scale.z = 0.02
        
        self.m.pose.orientation.x = 0
        self.m.pose.orientation.y = 0
        self.m.pose.orientation.z = 0
        self.m.pose.orientation.w = 1
        #print( "points length: ", len(self.m.points))
        self.path_pub.publish(self.m)   

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

        # Min distance to goal
        self.threshold_distance = 0.001

        # degree of freedom
        self.dof            = 4

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
        print ("yaw: ", self.yaw)
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
        # print ("JB: ", JB)
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
        thetaExt    = np.concatenate([np.array([deg90,                      0]),    np.array([theta])])
        dExt        = np.concatenate([np.array([self.bmz,            self.bmx]),    np.array([0])])
        aExt        = np.concatenate([np.array([0,                          0]),    np.array([self.l])])
        alphaExt    = np.concatenate([np.array([deg90,                 -deg90]),    np.array([0])])

        self.T      = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)

        T  = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)
        JB = jacobian(T, self.revolute + [True])
        
        return JB
        
    
    def get_error(self):
        # Calculate the end effector pose
        x, y, z, yaw = self.x, self.y, self.z, self.yaw
        
        # Calculate the error between the desired and current end-effector pose

        error = np.array([[self.d_sigma[0] - x], [self.d_sigma[1] - y], [self.d_sigma[2] - z], [normalize_angle(self.d_sigma[3] - float(self.yaw))] ])
        # error = np.array([[self.d_sigma[0] - x], [self.d_sigma[1] - y], [self.d_sigma[2] - z], [0]])

        print ("error: ", error)
        return error
    
    
if __name__ == "__main__":

    ros_node = TP_controller()      
    rospy.spin()