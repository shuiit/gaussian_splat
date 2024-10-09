
import numpy as np
import scipy.io
import pandas as pd

class Points3D():
        def __init__(self,point_3d,real_coord,frame_number):
            self.points_3d = point_3d
            self.real_coord = real_coord
            self.get_frame(frame_number)
            self.idx_to_real()


        def get_frame(self,frame_number):
            self.frame = frame_number
            self.points_in_idx = pd.concat([self.points_3d[body_wing][self.points_3d[body_wing]['frame'] == frame_number] for body_wing in ['body','rwing','lwing']])
            self.real_coord_frame = self.real_coord[self.real_coord[:,3] == frame_number,:]

        def idx_to_real(self):
            self.points_in_ew_frame = np.array([self.real_coord_frame[self.points_in_idx[ax] - 1,idx] for idx,ax in enumerate(['X','Y','Z'])]).T
            self.points_in_ew_frame_homo = np.hstack((self.points_in_ew_frame,np.ones([self.points_in_ew_frame.shape[0],1])))
            self.points_in_ew_frame  = np.column_stack((self.points_in_ew_frame,np.arange(self.points_in_ew_frame.shape[0])))

        def z_buffer(self,image,croped_camera_matrix = False):
            voxels_cam = np.matmul(image.world_to_cam, self.points_in_ew_frame_homo.T).T
            projected = image.project_on_image(self.points_in_ew_frame_homo,croped_camera_matrix)
            pxls = np.round(projected) 
            idx_sorted_by_z = voxels_cam[:,2].argsort()
            voxels_sorted_by_z = self.points_in_ew_frame[idx_sorted_by_z,:]
            [pixels,idx] = np.unique(pxls[idx_sorted_by_z,:], axis=0,return_index=True)
            return voxels_sorted_by_z[idx,:],pixels



        
              
