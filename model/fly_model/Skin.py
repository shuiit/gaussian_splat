
import os.path
import open3d as o3d
import numpy as np
import Plotter 


class Skin():
    def __init__(self, parts,path_to_mesh, scale = 1):
        self.parts = {part : idx for idx,part in enumerate(parts)}
        self.scale = scale
        self.load_skin(path_to_mesh)
        

    def load_skin(self,path_to_mesh):
        ptcloud_parts = {part : self.load_mesh(f'{path_to_mesh}/{part}.stl',idx) for part,idx in self.parts.items() if os.path.isfile(f'{path_to_mesh}/{part}.stl')}
        skin = np.vstack(list(ptcloud_parts.values()))
        self.ptcloud_skin = skin[:,0:3]*self.scale
        self.skin_normals = (skin[:,3:-1].T/np.linalg.norm(skin[:,3:-1], axis = 1)).T
        self.ptcloud_part_idx = skin[:,-1]

    def load_mesh(self,path_to_mesh,idx_mesh):
        mesh = o3d.io.read_triangle_mesh(path_to_mesh)
        pt_cloud = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        idx_of_part = np.ones((pt_cloud.shape[0],1))*idx_mesh

        return np.hstack((pt_cloud,normals,idx_of_part))
    
    def translate_ptcloud_skin(self,translation):
        self.ptcloud_skin = self.ptcloud_skin - translation


    def calculate_weights(self,skeleton,constant_weight = [False,'right_wing_root','left_wing_root'],**kwargs):
        self.weight = np.vstack([skeleton.calculate_weight(self.get_part(part), constant_weight = constant_weight,**kwargs) for constant_weight,part in zip(constant_weight,self.parts.keys())])


    def rotate_skin(self,skeleton):
        
        points_homo = np.column_stack([self.ptcloud_skin,np.ones(self.ptcloud_skin.shape[0])])
        normals_homo = np.column_stack([self.skin_normals,np.ones(self.skin_normals.shape[0])])

        # for every bone, rotate the skin to the joint coordinate systm (in bind position), (each bone is defined by the coordinates of the parent joint)
        # then rotate to the new position by multiplying by global_transformation (transforming from local joint coordinates to global) 
        rotated_points = [skeleton.joints[joint_name].rotate_to_new_position(weight[:,np.newaxis],points_homo) for weight,joint_name in zip(self.weight.T,skeleton.bones)]
        normals_rotated = [skeleton.joints[joint_name].rotate_normal_to_new_position(weight[:,np.newaxis],normals_homo) for weight,joint_name in zip(self.weight.T,skeleton.bones)]
        normals_rotated = np.sum(normals_rotated,axis = 0)[:,0:3]
        
        
        
        return np.sum(rotated_points,axis = 0)[:,0:3],(normals_rotated.T/np.linalg.norm(normals_rotated, axis = 1)).T


    def get_part(self,part):
        return self.ptcloud_skin[self.ptcloud_part_idx == self.parts[part],:]
    
    # def translate_skin_by_bone(self, parent_joint,points_homo,skip = 10):


    # def calculate_weight(self,skeleton, idx):
    #     skeleton.calculate_weight(body)

    


    
