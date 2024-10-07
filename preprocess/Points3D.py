
import numpy as np
import scipy.io
import pandas as pd
import numpy as np

class Points3D():
        def __init__(self,path):
            self.path = path
            self.points_3d = {body_wing : pd.DataFrame(self.load_hull(body_wing),columns = ['X','Y','Z','frame']) for body_wing in ['body','rwing','lwing']}
            self.real_coord = scipy.io.loadmat(f'{self.path}/3d_pts/real_coord.mat')['all_coords']

        def load_hull(self,body_wing):
            return scipy.io.loadmat(f'{self.path}/3d_pts/{body_wing}.mat')['hull']

        def get_frame(self,frame_number):
            self.frame = frame_number
            self.points_in_idx = pd.concat([self.points_3d[body_wing][self.points_3d[body_wing]['frame'] == frame_number] for body_wing in ['body','rwing','lwing']])
            self.real_coord_frame = self.real_coord[self.real_coord[:,3] == frame_number,:]

        def idx_to_real(self):
            self.points_in_ew_frame = np.array([self.real_coord_frame[self.points_in_idx[ax] - 1,idx] for idx,ax in enumerate(['X','Y','Z'])]).T
              
