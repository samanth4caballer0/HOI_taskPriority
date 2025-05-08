# ros node that publishes the goal position and orientation

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np

class ArmGoalPublisher:
    def __init__(self):

        rospy.init_node("ArmGoalPublisher")
        rospy.loginfo("Starting ArmGoalPublisher ....")

        self.goalPublisher = rospy.Publisher('/desired_sigma', PoseStamped, queue_size=10)
        
        self.goal_vector = [[0.2, 0.0, -0.25, 0.1],[0.3, 0.2, -0.16, 0.5], [-0.3, -0.2, -0.34, 0.05],[0.1, 0.0, -0.3, -0.3]]

        rospy.Timer(rospy.Duration(20), self.publishgoal)
 
        
    def publishgoal(self,_):
        if len(self.goal_vector) > 0:
            print ("Publishing goal: ", self.goal_vector[0])
            goal = self.goal_vector.pop(0)
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'world_ned'
            pose.pose.position.x = goal[0]
            pose.pose.position.y = goal[1]
            pose.pose.position.z = goal[2]            

            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = np.sin(goal[3]/2)
            pose.pose.orientation.w = np.cos(goal[3]/2)
            
            self.goalPublisher.publish(pose)

    
if __name__ == "__main__":

    ros_node = ArmGoalPublisher()      
    rospy.spin()