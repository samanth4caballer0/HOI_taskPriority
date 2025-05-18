# ros node that publishes the goal position and orientation

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import numpy as np

class ArmGoalPublisher:
    def __init__(self):

        rospy.init_node("ArmGoalPublisher")
        rospy.loginfo("Starting ArmGoalPublisher ....")
        self.goal = None
        self.goalPublisher = rospy.Publisher('/desired_sigma', PoseStamped, queue_size=10)

        rospy.Subscriber('/goal_request', Bool, self.goal_request_callback)
        

 
    def goal_request_callback(self, msg):
        if msg.data:
            
            print ("Received goal request")
            goal = np.random.uniform(low=-0.3, high=0.3, size=(1,))                  # X
            goal = np.append(goal, np.random.uniform(low=-0.3, high=0.3))           # Y
            goal = np.append(goal, np.random.uniform(low=-0.15, high=-0.35))         # Z
            goal = np.append(goal, np.random.uniform(low=-np.pi/2, high=np.pi/2))   # Yaw
            print ("Publishing goal: ", goal)

            
            if  self.goal is not None and np.linalg.norm(self.goal-goal) < 0.9 :
                pass
            else:
                self.goal = goal
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