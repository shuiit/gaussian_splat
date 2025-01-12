
import sys
import os

splat_dir = 'D:/Documents/gaussian_splat/model/'

sys.path.insert(0, f'{splat_dir}/fly_model')
import numpy as np
from Skin import Skin
import plotly.graph_objects as go
from Joint import Joint
parent_dir = os.path.abspath(os.path.join(os.getcwd(),'camera_frames'))
sys.path.insert(0, f'{splat_dir}/camera_frames')
from Frame import Frame

import numpy as np
from Skin import Skin
import plotly.graph_objects as go
import Utils



# initilize skeletone, joints and bones
# body angles - yaw,pitch,roll
# wing angles - phi, psi, theta
class Model():
    def __init__(self,path_to_mesh, path_to_frame,skeleton_scale = 1/1000, constant_weight = False, color = 'green'):
        self.path_to_mesh = path_to_mesh
        self.path_to_frame = path_to_frame
        self.skeleton_scale = skeleton_scale
        self.skin_translation = np.array([-0.1-1,0,1])*skeleton_scale
        

        self.initilize_skeleton()
        self.initilize_ptcloud()
        self.initilize_joints()
        self.find_2dcm_from_projection()
        joint_to_update = [self.root,self.thorax]
        rotation = [[0,-25,0],[0,-10,0]]
        translation = [self.cm_point_lab,self.thorax.translation_from_parent]
        [self.update_local_rotation(joint_to_update,rotation) for joint_to_update,rotation in zip(joint_to_update,rotation)]
        [self.update_local_translation(joint_to_update,translation) for joint_to_update,translation in zip(joint_to_update,translation)]
        self.global_rotated,self.global_normal = self.update_skin_and_joints()

        self.global_rotated_ew = self.rotate_to_ew(self.global_rotated)
        self.global_normal_ew = self.rotate_to_ew(self.global_normal)





    def initilize_skeleton(self,pitch_body = 0):

        self.root = Joint([1,0,0],[0,-pitch_body,0],parent = None, end_joint_of_bone = False, scale = self.skeleton_scale )
        self.neck = Joint([0.6,0,0.3],[0,pitch_body,0],parent = self.root, end_joint_of_bone = False, scale = self.skeleton_scale )
        self.neck_thorax =  Joint([0.6,0,0.3],[0,-25,0], parent = self.root, end_joint_of_bone = False, scale = self.skeleton_scale )
        head  =Joint([0.3,0,0],[0,0,0], parent = self.neck, scale = self.skeleton_scale )
        self.thorax  =Joint([-1,0,0],[0,25,0], parent= self.neck_thorax ,scale = self.skeleton_scale )
        abdomen = Joint([-1.3,0,0],[0,0,0], parent = self.thorax, scale = self.skeleton_scale )
        right_sp_no_bone = Joint([0,0,0.3],[0,pitch_body,0],parent = self.root, end_joint_of_bone = False , scale = self.skeleton_scale , color = 'red', rotation_order = 'zxy')
        self.right_wing_skeleton_root = Joint([0,-0.3,0],[0,0,0], parent = right_sp_no_bone, end_joint_of_bone = False, scale = self.skeleton_scale , color = 'red',rotation_order = 'zxy')
        right_wing_tip = Joint([0,-2.2,0],[0,0,0], parent = self.right_wing_skeleton_root, scale = self.skeleton_scale , color = 'red',rotation_order = 'zxy')
        left_sp_no_bone = Joint([0,0,0.3],[0,pitch_body,0], parent = self.root, end_joint_of_bone = False, scale = self.skeleton_scale , color = 'blue',rotation_order = 'zxy')
        self.left_wing_skeleton_root = Joint([0,0.3,0],[0,0,0],parent = left_sp_no_bone, end_joint_of_bone = False, scale = self.skeleton_scale , color = 'blue',rotation_order = 'zxy')
        left_wing_tip = Joint([0,2.2,0],[0,0,0], parent = self.left_wing_skeleton_root, scale = self.skeleton_scale , color = 'blue',rotation_order = 'zxy')
        self.joint_list = self.root.get_list_of_joints()
        self.list_joints_pitch_update = [self.neck,right_sp_no_bone,left_sp_no_bone]

        

    def initilize_ptcloud(self):
        self.body_skin = Skin(f'{self.path_to_mesh}/body.stl',scale = 1,color = 'lime')
        self.right_wing_skin = Skin(f'{self.path_to_mesh}/right_wing.stl',scale = 1,constant_weight = self.right_wing_skeleton_root,color = 'crimson')
        self.left_wing_skin = Skin(f'{self.path_to_mesh}/left_wing.stl',scale = 1, constant_weight = self.left_wing_skeleton_root,color = 'dodgerblue')
        self.fly_skin = Skin(f'{self.path_to_mesh}/body.stl',scale = 1,color = 'lime')
        self.all_skin = [self.body_skin, self.right_wing_skin,self.left_wing_skin,self.fly_skin]


    def initilize_joints(self):
        # initilize joints
        joints_of_bone = self.root.get_and_assign_bones()
        [skin.add_bones(joints_of_bone) for skin in  self.all_skin]
        [skin.translate_ptcloud_skin(self.skin_translation) for skin in  self.all_skin]
        self.body_skin.calculate_weights_dist(self.body_skin.bones[0:3])
        self.right_wing_skin.calculate_weights_constant()
        self.left_wing_skin.calculate_weights_constant()

        self.fly_skin.ptcloud_skin = np.vstack([self.body_skin.ptcloud_skin,self.right_wing_skin.ptcloud_skin,self.left_wing_skin.ptcloud_skin])



        
        self.color = np.full(self.fly_skin.ptcloud_skin.shape,100)
        self.fly_skin.skin_normals = np.vstack([self.body_skin.skin_normals,self.right_wing_skin.skin_normals,self.left_wing_skin.skin_normals])

        self.fly_skin.weights = np.vstack([self.body_skin.weights,self.right_wing_skin.weights,self.left_wing_skin.weights])
    
        
        # all_skin_points = [skin.rotate_skin_points() for skin in self.all_skin]
        # all_skin_normals = [skin.rotate_skin_normals() for skin in self.all_skin]
        
    

    def find_2dcm_from_projection(self):
        # load frames and cameras
        path = self.path_to_frame
        frames = list(range(900,910,1))
        image_name= []
        for frame in frames:
            image_name += [f'P{frame}CAM{cam + 1}' for cam in range(4)]

        frames = {f'{im_name}.jpg':Frame(path,im_name,idx) for idx,im_name in enumerate(image_name)}

        frame_number = 900 
        frame_names = ['P900CAM1.jpg','P900CAM2.jpg','P900CAM3.jpg','P900CAM4.jpg']
        frame_names = [f'P{frame_number}CAM{idx}.jpg' for idx in range(1,5)]

        camera_pixel = np.vstack([frames[frame].camera_center_to_pixel_ray(frames[frame].cm) for frame in  frame_names])
        camera_center = np.vstack([frames[frame].X0.T for frame in  frame_names])
        self.rot_mat_ew_to_lab = frames['P900CAM1.jpg'].rotation_matrix_from_vectors(frames['P900CAM1.jpg'].R[2,:], [0,0,1])
        cm_point = Utils.triangulate_least_square(camera_center,camera_pixel)
        self.cm_point_lab = np.squeeze(np.dot(self.rot_mat_ew_to_lab,cm_point[:,np.newaxis]).T)


    # joint_to_update.set_local_translation(cm_point_lab + np.array([-0.2,-0.4,0.22])*1/1000)
    # root_no_bone.set_local_rotation([0,pitch,0])
    def update_local_rotation(self, joint_to_update,rotation):
        joint_to_update.set_local_rotation(rotation) 
        if joint_to_update == self.root:
            [joint.set_local_rotation([0,-rotation[1],0]) for joint in self.list_joints_pitch_update]


    def update_local_translation(self,joint_to_update,translation):
        joint_to_update.set_local_translation(translation) 


    def update_skin_and_joints(self):
        [joint.update_rotation() for joint in self.joint_list]
        return  self.fly_skin.rotate_skin_points(),self.fly_skin.rotate_skin_normals() 
        # all_skin_normals_parts = [skin.rotate_skin_normals() for skin in self.all_skin]


    def rotate_to_ew(self,points):
        return np.dot(self.rot_mat_ew_to_lab.T,points.T).T




