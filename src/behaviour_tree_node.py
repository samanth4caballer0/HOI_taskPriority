#!/usr/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped
import time
from tf.transformations import quaternion_from_euler
from std_srvs.srv import SetBool
import actionlib

import py_trees
import py_trees.decorators
import py_trees.display
from py_trees.blackboard import Blackboard

from HOI_taskPriority.msg import TaskMsg
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
import operator 
import tf
import numpy as np
from tasks import *


class ScanObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ScanObject, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.READ)
        
        self.detect_goal = False

    def setup(self):
        self.logger.debug("  %s [ScanObject::setup()]" % self.name)

        self.logger.debug("  %s [ScanObject::setup() SUCCESS]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [ScanObject::initialise()]" % self.name)
        # SUBSCRIBERS
        # Subcribe to get aruco pose
        self.aruco_pose_object_sub  = rospy.Subscriber("/aruco_pose", PoseStamped, self.arucoPoseCB)
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)

        time.sleep(0.2)

    def update(self):
        task_msg = TaskMsg()
        task_msg.ids = "0"
        task_msg.name = "ScanObject"
        self.task_publisher.publish(task_msg)

        if self.detect_goal == True:
            time.sleep(1.0)
            self.logger.debug("  %s [ScanObject::Update() SUCCESS]" % self.name)
            self.blackboard.goal = self.aruco_pose.copy()
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug("  %s [ScanObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [ScanObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))

    # Aruco pose detector callback
    def arucoPoseCB(self, arucoPose):
        obj_pos_x = float(arucoPose.pose.position.x)
        obj_pos_y = float(arucoPose.pose.position.y)
        obj_pos_z = float(arucoPose.pose.position.z)

        self.aruco_pose = [obj_pos_x, obj_pos_y, obj_pos_z]# [1.18, 0.02]
        self.detect_goal = True

#behavio to make the arm turn towards the object
class FaceObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(FaceObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "orient_goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            "orient_goal", access=py_trees.common.Access.READ)
    
    def setup(self):
        self.logger.debug("  %s [FaceObject::setup()]" % self.name)
        self.err = np.array([np.inf, np.inf])
        self.logger.debug("  %s [FaceObject::setup() SUCCESS]" % self.name)
        
    def initialise(self):
        self.logger.debug("  %s [FaceObject::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)

        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        time.sleep(1.0)
    
    def update(self):
        #task error callbacj in behavior has dimension 1, as well as the desired. then the task is 3 instead of 1 
        task_msg = TaskMsg()
        task_msg.ids = "3"
        task_msg.name = "FaceObject"
        task_msg.desired = [np.pi/2.0]
        self.task_publisher.publish(task_msg)

        if  np.linalg.norm(self.err)< 0.1:
            self.logger.debug("  %s [FacingObject::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug("  %s [FacingObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [FacingObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        self.err = np.array([err.data[0]])    

class ApproachBasetoObject(py_trees.behaviour.Behaviour):               ####TODO fix desired poition!! to mae it a bit beforereaching he box 
    def __init__(self, name):
        super(ApproachBasetoObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.WRITE)                #TODO is this correct? or should i put another goal 
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.READ)
        
    def setup(self):
        self.logger.debug("  %s [ApproachBaseObject::setup()]" % self.name)
        self.err = np.array([np.inf, np.inf])
        self.logger.debug("  %s [ApproachBaseObject::setup() SUCCESS]" % self.name)
        
    def initialise(self):
        self.logger.debug("  %s [ApproachBaseObject::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)

        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        time.sleep(1.0)

    def update(self):
        task_msg = TaskMsg()
        task_msg.ids = "2"
        task_msg.name = "ApproachBasetoObject"
        task_msg.desired = [self.blackboard.goal[0], self.blackboard.goal[1]]       #subtract from each (x and Y) a distance of like 15cm 
        self.task_publisher.publish(task_msg)
        print(self.err)
        if  abs(self.err[0]) < 0.45 and abs(self.err[1]) < 0.45:
            self.logger.debug("  %s [ApproachBaseObject::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else: 
            self.logger.debug("  %s [ApproachBaseObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [ApproachBaseObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        if len(err.data) == 2:
            self.err = np.array([err.data[0], err.data[1]])    

class ApproachEEtoObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ApproachEEtoObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.READ)
        

    def setup(self):
        self.logger.debug("  %s [ApproachEEtoObject::setup()]" % self.name)
        self.err = np.array([np.inf, np.inf,  np.inf])
        self.logger.debug("  %s [ApproachEEtoObject::setup() SUCCESS]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [ApproachEEtoObject::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)

        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        time.sleep(1.0)
           
    def update(self):
        
        task_msg = TaskMsg()
        task_msg.ids = "1"
        task_msg.name = "ApproachEEtoObject"
        task_msg.desired = [self.blackboard.goal[0], self.blackboard.goal[1], -0.35]     #-0.3 why? 
        self.task_publisher.publish(task_msg)
        print(np.linalg.norm(self.err))
        if  np.linalg.norm(self.err)< 0.06:
            self.logger.debug("  %s [ApproachEEtoObject::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug("  %s [ApproachEEtoObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [ApproachEEtoObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        if len(err.data) == 3:
            self.err = np.array([err.data[0], err.data[1], err.data[2]])    
            
class LowerEEToObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(LowerEEToObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.READ)
        
    def setup(self):
        self.logger.debug("  %s [LowerEEtoObject::setup()]" % self.name)

        self.err = np.array([np.inf, np.inf, np.inf])
        self.logger.debug("  %s [LowerEEtoObject::setup() SUCCESS]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [LowerEEtoObject::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)

        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        time.sleep(1.0)
           
    def update(self):
        
        task_msg = TaskMsg()
        task_msg.ids = "1"
        task_msg.name = "LowerEEtoObject"
        task_msg.desired = [self.blackboard.goal[0], self.blackboard.goal[1], -0.150]   #-0.153
        self.task_publisher.publish(task_msg)
        print(self.err)
        if  self.err[2] < 0.004:
            self.logger.debug("  %s [LowerEEtoObject::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug("  %s [LowerEEtoObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [LowerEEtoObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        if len(err.data) == 3:
            self.err = np.array([err.data[0], err.data[1], err.data[2]])    

class EnableSuction (py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(EnableSuction, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [EnableSuction::setup()]" % self.name)
    
    def initialise(self):
        self.logger.debug("  %s [EnableSuction::initialise()]" % self.name)

    def update(self):
        succes = self.enable_suction()
        if succes:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
        
    def enable_suction(self):
        rospy.logwarn("Calling enable suction")
        rospy.wait_for_service('/turtlebot/swiftpro/vacuum_gripper/set_pump')
        path = []
        try:
            enable_suction = rospy.ServiceProxy('/turtlebot/swiftpro/vacuum_gripper/set_pump', SetBool)
            resp = enable_suction(True)
            
            return resp.success
        
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False

class PickupObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(PickupObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.READ)
        

    def setup(self):
        self.logger.debug("  %s [PickupObject::setup()]" % self.name)
        self.err = np.array([np.inf, np.inf,np.inf])
        self.logger.debug("  %s [PickupObject::setup() SUCCESS]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [PickupObject::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)
        task_msg = TaskMsg()
        task_msg.ids = "1"
        task_msg.name = "PickupObject"
        task_msg.desired = [self.blackboard.goal[0], self.blackboard.goal[1], -0.33]     #-0.3 why? 
        self.task_publisher.publish(task_msg)
        time.sleep(1.0)
        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        
           
    def update(self):
        
        task_msg = TaskMsg()
        task_msg.ids = "1"
        task_msg.name = "PickupObject"
        task_msg.desired = [self.blackboard.goal[0], self.blackboard.goal[1], -0.3]     #-0.3 why? 
        self.task_publisher.publish(task_msg)
        print(np.linalg.norm(self.err))
        if  self.err[2] < 0.004:    #np.linalg.norm(self.err)< 0.003:    
            self.logger.debug("  %s [PickupObject::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug("  %s [PickupObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [PickupObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        if len(err.data) == 3:
            self.err = np.array([err.data[0], err.data[1], err.data[2]])    

class HandleObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(HandleObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "orient_goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            "orient_goal", access=py_trees.common.Access.READ)
    
    def setup(self):
        self.logger.debug("  %s [HandleObject::setup()]" % self.name)
        self.err = np.array([np.inf])
        self.logger.debug("  %s [HandleObject::setup() SUCCESS]" % self.name)
        
    def initialise(self):
        self.logger.debug("  %s [HandleObject::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)

        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        time.sleep(1.0)
    
    def update(self):
        #task error callbacj in behavior has dimension 1, as well as the desired. then the task is 3 instead of 1 
        task_msg = TaskMsg()
        task_msg.ids = "3"
        task_msg.name = "HandleObject"
        task_msg.desired = [-np.pi/2.0] #or is it np.pi
        self.task_publisher.publish(task_msg)

        if  np.linalg.norm(self.err)< 0.1:
            self.logger.debug("  %s [HandleObject::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug("  %s [HandleObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [HandleObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        if len(err.data) == 1:

            self.err = np.array([err.data[0]])

class ApproachGoal(py_trees.behaviour.Behaviour):               ####TODO fix desired poition!! to mae it a bit beforereaching he box 
    def __init__(self, name):
        super(ApproachGoal, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.WRITE)                #TODO is this correct? or should i put another goal 
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.READ)
        
    def setup(self):
        self.logger.debug("  %s [ApproachGoal::setup()]" % self.name)
        self.err = np.array([np.inf, np.inf])
        self.logger.debug("  %s [ApproachGoal::setup() SUCCESS]" % self.name)
        
    def initialise(self):
        self.logger.debug("  %s [ApproachGoal::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)

        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        time.sleep(1.0)

    def update(self):
        task_msg = TaskMsg()
        task_msg.ids = "2"
        task_msg.name = "ApproachGoal"
        task_msg.desired = [0,0]       #subtract from each (x and Y) a distance of like 15cm 
        self.task_publisher.publish(task_msg)
        
        if  abs(self.err[0]) < 0.4 and abs(self.err[1]) < 0.4:
            self.logger.debug("  %s [ApproachGOal::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else: 
            self.logger.debug("  %s [ApproachGOal::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [ApproachGOal::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        if len(err.data) == 2:
            self.err = np.array([err.data[0], err.data[1]])    

class ReturnObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ReturnObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "orient_goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            "orient_goal", access=py_trees.common.Access.READ)
    
    def setup(self):
        self.logger.debug("  %s [ReturnObject::setup()]" % self.name)
        self.err = np.array([np.inf, np.inf])
        self.logger.debug("  %s [ReturnObject::setup() SUCCESS]" % self.name)
        
    def initialise(self):
        self.logger.debug("  %s [ReturnObject::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher("/task", TaskMsg, queue_size=10)

        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        time.sleep(1.0)
    
    def update(self):
        #task error callbacj in behavior has dimension 1, as well as the desired. then the task is 3 instead of 1 
        task_msg = TaskMsg()
        task_msg.ids = "3"
        task_msg.name = "ReturnObject"
        task_msg.desired = [np.pi/2.0] #or is it np.pi
        self.task_publisher.publish(task_msg)

        if  np.linalg.norm(self.err)< 0.1:
            self.logger.debug("  %s [ReturnObject::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug("  %s [ReturnObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [ReturnObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        self.err = np.array([err.data[0]])

class LetObject (py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(LetObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "n_object", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            "n_object", access=py_trees.common.Access.WRITE)
        self.blackboard.n_object = 1
    def setup(self):
        self.logger.debug("  %s [LetObject::setup()]" % self.name)
    
    def initialise(self):
        self.logger.debug("  %s [LetObject::initialise()]" % self.name)

    def update(self):
        succes = self.disable_suction()
        if succes:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
        
    def disable_suction(self):
        rospy.logwarn("Calling enable suction")
        rospy.wait_for_service('/turtlebot/swiftpro/vacuum_gripper/set_pump')
        path = []
        try:
            enable_suction = rospy.ServiceProxy('/turtlebot/swiftpro/vacuum_gripper/set_pump', SetBool)
            resp = enable_suction(False)
            
            return resp.success
        
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
        
#   Create Behavior trees function
def create_tree():
    # Special py_trees behavior
    # Check number of object the robot already went to
    n_object_lt_1 = py_trees.behaviours.CheckBlackboardVariableValue(
        name="n_object_lt_1",
        check=py_trees.common.ComparisonExpression(
            variable = "n_object",
            value = 1,
            operator=operator.lt
        )
    )

    # Create Behaviors
    scan_object = ScanObject(name="scan_object")
    face_object = FaceObject(name="face_object")
    approach_base_to_object = ApproachBasetoObject(name="approach_base_aruco_box")
    approach_ee_to_object = ApproachEEtoObject(name="approach_aruco_box")
    scan_object_again = ScanObject(name="scan_object_again")

    lower_ee_to_object = LowerEEToObject(name="lower_aruco_box")
    enable_suction = EnableSuction(name="enable_suction")
    pickup_object = PickupObject(name="pickup_object")
    handle_object = HandleObject(name="handle_object")
    approach_goal = ApproachGoal(name="approach_goal")
    return_object = ReturnObject(name="return_object")
    let_object = LetObject(name="let_object")
    
    # pick_object = PickObject(name="pick_object")

    # approach_base_to_place = ApproachBasePlace(name="approach_base_place")

    # handle_manipulator_object= HandleManipulatorObject(name="handle_manipulator_object")

    # approach_manipulator_to_place_object = ApproachManipulatorPlaceObject(name="approach_manipulator_place_object")

    # let_object = LetObject(name="let_object")

    root = py_trees.composites.Sequence(name="Life", memory=True)    
    root.add_children([
                       scan_object,
                        face_object,
                       approach_base_to_object,
                       scan_object_again,
                       approach_ee_to_object,
                       lower_ee_to_object,
                          enable_suction,
                            pickup_object,
                            handle_object,
                            approach_goal,
                            return_object,
                            let_object,
                       ])
    # py_trees.display.render_dot_tree(root)
    return root

def run(it=10000):
    root = create_tree()

    try:
        print("Call setup for all tree children")
        root.setup_with_descendants() 
        print("Setup done!\n\n")
        py_trees.display.ascii_tree(root)
        
        for _ in range(it):
            root.tick_once()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    py_trees.logging.level = py_trees.logging.Level.DEBUG
   
    rospy.init_node('behavior_trees')

    # Create behavior tree
    root = create_tree()
    run()