# ros node that publishes the goal position and orientation

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np

class ArmGoalPublisher:
    def __init__(self):

        rospy.init_node("ArmGoalPublisher")
        rospy.loginfo("Starting ArmGoalPublisher ....")

        self.goalPublisher = rospy.Publisher('/desired_sigma', PoseStamped, queue_size=10)
        
        self.goal_vector = [[0.5, 0.5, 0.5, 0],[0.2, 0.2, 0.2, 0],[0.3, 0.3, 0.3, np.pi]]

        rospy.Timer(rospy.Duration(0.1), self.publishgoal)

        
    def publishgoal(self):
        if len(self.goal_vector) > 0:
            goal = self.goal_vector.pop(0)
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'swiftpro/manipulator_base_link'
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