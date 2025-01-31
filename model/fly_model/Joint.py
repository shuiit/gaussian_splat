import numpy as np
from Bone import Bone


class Joint:
    def __init__(self, translation, rotation,  parent = None, end_joint_of_bone = True, rotation_order = 'zyx', scale = 1,color = 'green'):
        self.child = []
        self.parent = parent
        self.local_angles = rotation
        self.local_translation = translation
        self.rotation_order = list(rotation_order)
        self.local_rotation = self.rotation_matrix(rotation[0],rotation[1],rotation[2])
        self.translation_from_parent = np.array(translation)*scale
        self.local_transformation = self.transformation_matrix()
        self.global_transformation = self.get_global_transformation(rest_bind = True)
        self.end_joint_of_bone = end_joint_of_bone
        self.get_global_point()
        self.bone = None
        self.color = color
        self.scale = scale
        self.update_child()
        # self.parent.child.append(self) if self.parent != None else []
        


    def update_child(self):
        if self.parent == None:
            return
        self.parent.update_child()
        
        if self not in self.parent.child:
            self.parent.child.append(self)
       

    def get_and_assign_bones(self, visited = None):
        visited = visited or set()
            
        if self in visited:
            return []
        visited.add(self)
        if self.end_joint_of_bone:
            self.parent.bone = Bone(self.parent,self )
        bones = [self.parent] if self.end_joint_of_bone else []
        for child in self.child: 
            bones += child.get_and_assign_bones(visited)
        return bones
    
    # def get_and_assign_bones(self, visited = None, bones = None):
    #     visited = visited or set()
    #     print(bones)
    #     bones = bones or []
    #     visited.add(self)
    #     if self.end_joint_of_bone:
    #         self.parent.bone = Bone(self.parent,self )
    #         bones.append(self.parent)
    #     for child in self.child: 
    #         child.get_and_assign_bones(visited,bones)
    #     return bones
    


    def get_list_of_joints(self, visited = None,joints = None):
        visited = visited or set()
        joints = joints or []
        joints.append(self)
        visited.add(self)

        for child in self.child: 
            if child not in visited:
                child.get_list_of_joints(visited,joints)
        return joints




    def set_local_rotation(self,angles):
        # angles (z,y,x) (yaw, pitch, roll)
        self.local_rotation = self.rotation_matrix(angles[0],angles[1],angles[2])
        self.local_transformation = self.transformation_matrix()

    
    def set_local_translation(self,translation):
        self.translation_from_parent = np.array(translation)
        self.local_transformation = self.transformation_matrix()

    def set_local_transformation(self):
        self.local_transformation = self.transformation_matrix()

    def get_global_transformation(self, rest_bind = False):
        if self.parent == None:
            return self.local_transformation
        self.global_transformation = self.parent.get_global_transformation(rest_bind = rest_bind)
        self.global_transformation = np.dot(self.global_transformation,self.local_transformation)
        if rest_bind == True:
            self.bind_transformation = self.global_transformation
        return self.global_transformation
    
    def get_global_point(self,point = [0,0,0,1]):
        if point == [0,0,0,1]:
            self.global_origin = np.dot(self.global_transformation,point)[0:3]
        return np.dot(self.global_transformation,point)[0:3]
        

    def rotate_to_new_position(self,weight,points_homo):
        transformation_rest = np.linalg.inv(self.bind_transformation)
        rotated_points = np.dot(transformation_rest,points_homo.T)
        return weight*np.dot(self.global_transformation,rotated_points).T

    def update_rotation(self):
        self.get_global_transformation()
        self.get_global_point()


    def rotate_normal_to_new_position(self,weight,normal):
        transformation_rest = np.linalg.inv(self.bind_transformation)
        transformation_rest_to_global = np.dot(self.global_transformation,transformation_rest).T
        rotated_points_inv = np.linalg.inv(transformation_rest_to_global)
        return weight*np.dot(rotated_points_inv,normal.T).T





    def rotation_matrix(self,yaw,pitch,roll):
        roll = roll*np.pi/180
        pitch = pitch*np.pi/180
        yaw = yaw*np.pi/180
        mat = {}
        mat['x'] = np.array([[1,0,0],[0 ,np.cos(roll),-np.sin(roll)],[0, np.sin(roll), np.cos(roll)]])
        mat['y'] = np.array([[np.cos(pitch),0,np.sin(pitch)],[0, 1,0],[-np.sin(pitch), 0, np.cos(pitch)]])
        mat['z'] = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0, 0, 1]])
 
        rotation_matrix = mat[self.rotation_order[0]] @ mat[self.rotation_order[1]]  @ mat[self.rotation_order[2]] 
        return rotation_matrix
    
    def transformation_matrix(self):
        return  np.vstack((np.column_stack((self.local_rotation,self.translation_from_parent)),[0,0,0,1]))
