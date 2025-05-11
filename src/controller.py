import numpy as np
import sympy as sp 
import ros

class Controller:
    
    def __init__(self, robot):
        self.robot = robot
        self.q = np.zeros(4)
        self.J = np.zeros((4, 4))
        
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
        J = task_vector.jacobian(q)        

        return J 
    
    def weighted_DLS(A, damping, Weight):
        '''
            Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

            Arguments:
            A (Numpy array): matrix to be inverted
            damping (double): damping factor

            Returns:
            (Numpy array): inversion of the input matrix
        '''
        return np.linalg.inv(Weight) @ A.T @ np.linalg.inv(A @ np.linalg.inv(Weight) @ A.T + damping**2 * np.eye(np.shape(A)[0], np.shape(A)[0]))

    def controller(self, q, qd, qdd):
        
        ### Recursive Task-Priority algorithm (w/set-based tasks)
        # The algorithm works in the same way as in Lab4. 
        # The only difference is that it checks if a task is active.
        # Initialize null-space projector
        P  = np.eye(self.robot.dof, self.robot.dof)
        
        # Initialize output vector (joint velocity)
        dq  = np.zeros((self.robot.dof, 1))
        for i in range(len(self.tasks)):      
            # Update task state
            self.tasks[i].update(self.robot)
            if self.tasks[i].isActive() != 0:
                # Compute augmented Jacobian
                Jbar    = self.tasks[i].J @ P 
                # Compute task velocity
                # Accumulate velocity
                dq      = dq + (Jbar, 0.01, self.weight_matrix) @ (self.tasks[i].isActive() * self.tasks[i].err - self.tasks[i].J @ dq) 
                # Update null-space projector
                P       = P - (Jbar, 0.0001, self.weight_matrix) @ Jbar  
            else:
                dq      = dq
                P       = P 
        return dq
    
    def vel_controller(self, q, qd, qdd):
        dq  = np.zeros((self.robot.dof, 1))
        
        desired_vel = np.zeros((self.robot.dof, 1))
        
        return dq
    
    def set_weightMatrix(self, value):
        self.weight_matrix = value
    
    
