
import os.path
import open3d as o3d
import numpy as np


class Skin():
    def __init__(self,path_to_mesh, scale = 1, constant_weight = False, color = 'green'):
        self.scale = scale
        self.load_skin(path_to_mesh)
        self.constant_weight = constant_weight
        self.color = color

    def add_bones(self,joints_of_bone):
        self.bones = joints_of_bone
        
    def load_skin(self,path_to_mesh):
        if os.path.isfile(f'{path_to_mesh}'):
            skin = self.load_mesh(f'{path_to_mesh}')  
        self.ptcloud_skin = skin[:,0:3]*self.scale
        self.skin_normals = (skin[:,3:].T/np.linalg.norm(skin[:,3:-1], axis = 1)).T

    def load_mesh(self,path_to_mesh):
        mesh = o3d.io.read_triangle_mesh(path_to_mesh)
        pt_cloud = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        return np.hstack((pt_cloud,normals))
    
    def translate_ptcloud_skin(self,translation):
        self.ptcloud_skin = self.ptcloud_skin - translation


    def calculate_weights_dist(self):        
        # weights = np.vstack([joint.bone.calculate_dist_from_bone(self.ptcloud_skin) for joint in self.bones]).T
        # self.weights = (1/weights)/np.sum((1/weights),1)[:,np.newaxis]
         
        weights = np.array([joint.bone.calculate_dist_from_bone(self.ptcloud_skin) for joint in self.bones[0:3]]).T
        idx = np.argmin(weights, axis=1)
        weights = np.zeros((weights.shape[0],len(self.bones)))
        weights[np.arange(weights.shape[0]), idx] = 1
        self.weights = weights


    def calculate_weights_constant(self):  
        self.weights = np.zeros((self.ptcloud_skin.shape[0],len(self.bones)))
        self.weights[:,self.bones.index(self.constant_weight)] = 1

    # def calculate_weights(self,skeleton,constant_weight = [False,'right_wing_root','left_wing_root'],**kwargs):
    #     self.weight = np.vstack([skeleton.calculate_weight(self.get_part(part,self.ptcloud_skin), constant_weight = constant_weight,**kwargs) for constant_weight,part in zip(constant_weight,self.parts.keys())])


    def rotate_skin_points(self):
        
        points_homo = np.column_stack([self.ptcloud_skin,np.ones(self.ptcloud_skin.shape[0])])
        # for every bone, rotate the skin to the joint coordinate systm (in bind position), (each bone is defined by the coordinates of the parent joint)
        # then rotate to the new position by multiplying by global_transformation (transforming from local joint coordinates to global) 
        rotated_points = [joint.rotate_to_new_position(weight[:,np.newaxis],points_homo) for weight,joint in zip(self.weights.T,self.bones)]
        return np.sum(rotated_points,axis = 0)[:,0:3]


    def rotate_skin_normals(self):
        
        normals_homo = np.column_stack([self.skin_normals,np.ones(self.skin_normals.shape[0])])
        # for every bone, rotate the skin to the joint coordinate systm (in bind position), (each bone is defined by the coordinates of the parent joint)
        # then rotate to the new position by multiplying by global_transformation (transforming from local joint coordinates to global) 
        normals_rotated = [joint.rotate_normal_to_new_position(weight[:,np.newaxis],normals_homo) for weight,joint in zip(self.weights.T,self.bones)]
        normals_rotated = np.sum(normals_rotated,axis = 0)[:,0:3]
        return (normals_rotated.T/np.linalg.norm(normals_rotated, axis = 1)).T


    def get_part(self,part,points):
        return points[self.ptcloud_part_idx == self.parts[part],:]
    
    # def translate_skin_by_bone(self, parent_joint,points_homo,skip = 10):


    # def calculate_weight(self,skeleton, idx):
    #     skeleton.calculate_weight(body)

    


    
