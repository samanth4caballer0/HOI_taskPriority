import numpy as np 

class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.erroVec = [] # error vector
        #self.l
        self.ff = None
        self.k = None

    def getFF(self):
        return self.ff
    
    def setFF(self, ff):
        self.ff = ff

    def getK(self):
        return self.k

    def setK(self, k):
        self.k = k
 
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err
    
    """
        Method that activation state of the task.
    """
    def isActive (self):
        return self.active

'''
    Subclass of Task, representing the 2D position task.
'''
class Position3D(Task):
    # Control the robot to reach a desired position in 3D cartesian space
    def __init__(self, name, robot, link):
        super().__init__(name,np.array([0.2, 0.0, -0.05]).reshape((3,1))) 
        self.J = np.zeros((3,robot.dof))                  # Initialize with proper dimensions
        self.err = np.zeros((3,1))                                  # Initialize with proper dimensions
        self.setK(np.eye(3))
        self.setFF(np.zeros((3,1)))
        self.link = link
        self.active = True
        self.goal_vector = [np.array([0.2, 0.0, -0.05]),
                            np.array([0.1, 0.2, -0.05]), 
                            np.array([0.1, -0.2, -0.05]),
                            np.array([0.1, 0.2, -0.2])]

        
    def update(self, robot):
        self.J = robot.get_armJacobian()[:3,:].reshape((3,self.link))                     # Update task Jacobian
        self.J = np.hstack((self.J, np.zeros((3, robot.dof - self.link))))
        current_position = np.array([[robot.x], [robot.y], [robot.z]]) # Compute current sigma

        self.err = np.array(self.getDesired() - current_position) # Update task error
        self.erroVec.append(np.linalg.norm(self.err))

    def setRandomDesired(self):
        #random = (np.random.rand(2,1)*4-2).reshape((2,1))
        
        return self.goal_vector.pop(0).reshape((3,1))

'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, robot, link):
        super().__init__(name, np.array([0.0]).reshape((1,1))) # desired orientation
        self.J = np.zeros((1,robot.dof))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        self.setK(np.eye(1))
        self.setFF(np.zeros((1,1)))
        self.link = link
        self.active = True

        
    def update(self, robot):
        #print ("jacobian: ", robot.get_armJacobian())
        self.J = robot.get_armJacobian()[3,:].reshape((1,self.link))   # Update task Jacobian
        self.J = np.pad(self.J, (0, robot.dof - self.link), mode='constant', constant_values=0)
        current_sigma =wrapangle(np.array([robot.q[0]+robot.q[3]]).reshape((1,1))) # Compute current sigma
        print ("current sigma: ",current_sigma*180/np.pi)
        self.err = wrapangle(self.getDesired() - current_sigma.reshape((1,1))) # Update task error
        self.error = np.array([-self.err]).reshape((1,1)) # Update task error
        print ("angular error in deg: ", self.err*180/np.pi)
        print ("base: ", robot.q[0]*180/np.pi)
        print ("cup: ", robot.q[3]*180/np.pi)
        self.erroVec.append(self.err[0])
        pass # to remove

    def setRandomDesired(self):
        self.setDesired( (np.random.rand(1,1)*2*np.pi-np.pi).reshape((1,1)))
        pass
"""
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, robot: Manipulator, link, desired = None):
        super().__init__(name, desired.pop(0))

        self.J = np.zeros((3,robot.getDOF()))# Initialize with proper dimensions
        self.err = np.zeros((3,1))# Initialize with proper dimensions
        self.erroVec = [[],[]]
        self.setK(np.eye(3))
        self.setFF(np.zeros((3,1)))
        self.link = link
        self.active = True
        self.desiredVector = desired
        
    def update(self, robot: Manipulator):
        positionJacobian = robot.getLinkJacobian(self.link)[:2,:].reshape((2,self.link))                     # Update task Jacobian
        positionJacobian = np.hstack((positionJacobian, np.zeros((2, robot.dof - self.link))))
        self.J[0:2,:] = positionJacobian # Update task Jacobian

        orientationJacobian = robot.getLinkJacobian(self.link)[5,:].reshape((1,self.link))   # Update task Jacobian
        orientationJacobian =  np.hstack((orientationJacobian, np.zeros((1, robot.dof - self.link))))
        self.J[2,:] = orientationJacobian

        current_transform = robot.getLinkTransform(self.link) # Compute current sigma
        current_sigma_angle = np.arctan2(current_transform[1,0], current_transform[0,0]) # Compute current sigma angle
        current_sigma_pos = current_transform[0:2,3] # Compute current sigma position
        error_pos = self.getDesired()[0:2] - current_sigma_pos.reshape((2,1)) # Compute position error
        error_angle = self.getDesired()[2] - current_sigma_angle

        self.err = np.array([error_pos[0], error_pos[1], error_angle]).reshape((3,1)) # Update task error
        self.erroVec[0].append(np.linalg.norm(error_pos))
        self.erroVec[1].append(error_angle[0])
        

    def setRandomDesired(self, random = True):
        if random or len(self.desiredVector) == 0:
            self.setDesired(np.array([np.random.rand(1,1)*4-2,np.random.rand(1,1)*4-2, np.random.rand(1,1)*2*np.pi-np.pi]).reshape((3,1)))
        else:
            self.setDesired(self.desiredVector.pop(0))
        print ("Desired position: ", self.getDesired())
        
''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired, robot: Manipulator, link):
        super().__init__(name, desired)
        self.link = link
        self.J = np.zeros((1,self.link))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        self.setK(np.eye(1))
        self.setFF(np.zeros((1,1)))


        
    def update(self, robot: Manipulator):
        self.J = robot.getLinkJacobian(self.link)[5,:].reshape((1,self.link))   # Update task Jacobian
        self.J = np.pad(self.J, (0, robot.dof - self.link), mode='constant', constant_values=0)
        current_sigma = robot.getLinkOrientation(self.link) # Compute current sigma
        print('current_sigma:',current_sigma)
        self.err = wrapangle(self.getDesired() - current_sigma.reshape((1,1))) # Update task error
        print ("angular error: ", self.err)
        self.erroVec.append(self.err[0])
        pass # to remove

    def setRandomDesired(self):
        self.setDesired( (np.random.rand(1,1)*2*np.pi-np.pi).reshape((1,1)))
        pass

''' 
    Subclass of Task, representing the Obstacle avoidance task.
'''
class Obstacle2D(Task):
    def __init__(self, name, position, thresholds, robot: Manipulator,):
        super().__init__(name, None)
        self.position = position
        self.activation_tresh = thresholds[0]
        self.deactivation_tresh = thresholds[1]
        self.J = np.zeros((2,robot.dof))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        self.active = False
        self.setK(np.eye(2))
        self.setFF(np.zeros((2,1)))
    
    def activate (self, sigma):
        if self.active == False and np.linalg.norm(sigma) <= self.activation_tresh:
            self.active = True
        elif self.active == True and np.linalg.norm(sigma) >= self.deactivation_tresh:
            self.active = False

    def update(self, robot: Manipulator):
        self.J = robot.getEEJacobian()[:2,:].reshape((2,robot.dof))   # Update task Jacobian
        current_sigma = robot.getEETransform()[:2,3].reshape((2,1)) - self.position#g get EE x & y
        self.activate(current_sigma)
        print("current_sigma: ",current_sigma)
        self.err = current_sigma/np.linalg.norm(current_sigma)
        print ("angular error: ", self.err)
        pass # to remove
"""
class JointLimit2D(Task)    :
    def __init__(self, name, link, limits, tresholds):
        super().__init__(name, None)
        self.link = link
        self.limits = limits
        self.activation_tresh = tresholds[0]
        self.deactivation_tresh = tresholds[1]
        self.J = np.zeros((1,self.link))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        self.setK(np.eye(1))
        self.setFF(np.zeros((1,1)))
        self.active = 0

    def update(self, robot):
        self.J = robot.get_armJacobian()[3,:].reshape((1,-1))   # Update task Jacobian
        self.J = np.pad(self.J, (0, robot.dof - self.link), mode='constant', constant_values=0)
        current_sigma = robot.q[self.link] # Compute current sigma
        self.activate(current_sigma)
        print('current_sigma:',current_sigma)
        self.err = np.array([self.active]) # Update task error
        print ("angular error: ", self.err)
        self.erroVec.append(self.err)
        pass # to remove

    
    def activate (self, angle):
        if self.active == 0 and angle >=  self.limits[0] - self.activation_tresh:
            self.active = -1
        elif self.active == 0 and angle <= self.limits[1] + self.activation_tresh:
            self.active = 1
        elif self.active == -1 and angle <= self.limits[0] - self.deactivation_tresh:
            self.active = 0
        elif self.active == 1 and angle >= self.limits[1] + self.deactivation_tresh:
            self.active = 0
    
    def isActive(self):
        return abs(self.active)


    def setRandomDesired(self):
        self.setDesired( (np.random.rand(1,1)*2*np.pi-np.pi).reshape((1,1)))
        pass
 

def wrapangle(angle):
    # Wrap angle to the range [-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi