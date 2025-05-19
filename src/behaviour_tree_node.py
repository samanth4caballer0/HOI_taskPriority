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

from .tasks import *
from HOI_taskPriority.msg import TaskMsg
from std_msgs.msg import Float64MultiArray
from config import *
from nav_msgs.msg import Odometry
import tf
import operator 


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
        self.aruco_pose_object_sub      = rospy.Subscriber(aruco_pose_topic, PoseStamped, self.arucoPoseCB)
        time.sleep(0.2)

    def update(self):            
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

class ApproachObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ApproachObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            "goal", access=py_trees.common.Access.READ)

    def setup(self):
        self.logger.debug("  %s [ApproachObject::setup()]" % self.name)
        self.err = np.array([np.inf, np.inf])
        self.logger.debug("  %s [ApproachObject::setup() SUCCESS]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [ApproachObject::initialise()]" % self.name)  
        # PUBLISHERS
        # Publisher for sending task to the TP control node
        self.task_publisher = rospy.Publisher(task_topic, TaskMsg, queue_size=10)

        # SUBSCRIBERS
        #subscriber to task error 
        self.task_err_sub = rospy.Subscriber("/task_error", Float64MultiArray, self.get_err) 

        # Wait 0.2s to init pub and sub
        time.sleep(1.0)
           
    def update(self):
        
        task_msg = TaskMsg()
        task_msg.ids = "2"
        task_msg.name = "ApproachObject"
        task_msg.desired = [self.blackboard.goal[0], self.blackboard.goal[1], -0.3]
        self.task_publisher.publish(task_msg)

        if  np.linalg.norm(self.err)< 0.05:
            self.logger.debug("  %s [ApproachBaseObject::Update() SUCCESS]" % self.name)
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug("  %s [ApproachBaseObject::Update() RUNNING]" % self.name)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug("  %s [ApproachBaseObject::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))
        
    def get_err(self, err):
        self.err = np.array([err.data[0], err.data[1], err.data[2]])    


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

    approach_base_to_object = ApproachBaseObject(name="approach_base_object")

    approach_manipulator_to_object = ApproachManipulatorObject(name="approach_manipulator_object")

    pick_object = PickObject(name="pick_object")

    approach_base_to_place = ApproachBasePlace(name="approach_base_place")

    handle_manipulator_object= HandleManipulatorObject(name="handle_manipulator_object")

    approach_manipulator_to_place_object = ApproachManipulatorPlaceObject(name="approach_manipulator_place_object")

    let_object = LetObject(name="let_object")

    root = py_trees.composites.Sequence(name="Life", memory=True)    
    root.add_children([n_object_lt_1,
                       scan_object,
                       approach_base_to_object, 
                       approach_manipulator_to_object, 
                       pick_object, 
                       handle_manipulator_object, 
                       approach_base_to_place, 
                       approach_manipulator_to_place_object, 
                       let_object])
    # py_trees.display.render_dot_tree(root)
    return root

def run(it=200):
    root = create_tree()

    try:
        print("Call setup for all tree children")
        root.setup_with_descendants() 
        print("Setup done!\n\n")
        py_trees.display.ascii_tree(root)
        
        for _ in range(it):
            root.tick_once()
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    py_trees.logging.level = py_trees.logging.Level.DEBUG
   
    rospy.init_node('behavior_trees')

    # Create behavior tree
    root = create_tree()
    run()