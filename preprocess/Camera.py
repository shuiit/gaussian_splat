
import numpy as np
import scipy.io

class Camera():
    def __init__(self,path,camera_number):
        self.path = path
        cam = scipy.io.loadmat(f'{path}/camera_KRX0.mat')
        self.K = cam['camera'][0:3,0:3,camera_number]
        self.R = cam['camera'][0:3,3:6,camera_number]
        self.X0 = cam['camera'][0:3,6:7,camera_number]
        self.t = np.matmul(self.R,self.X0)
        self.camera_number = camera_number + 1
        self.K[1,2]  = self.K[1,2] - 1  
        self.K[0,2]  = self.K[0,2] - 1
        self.camera_matrix = np.hstack([np.matmul(self.K,self.R),-np.matmul(self.K,self.t)])
        self.rotmat2qvec()
        


    def camera_calibration_crop(self,crop_pixels):
        """updates the intrinsic K matrix for croped images

        Args:
            crop_pixels (np array): loaction of left bottom pixel (we need the bottom because we flip it) [x,y]
        """
        self.K_crop = self.K.copy()
        self.K_crop[0,2] = self.K[0,2] - crop_pixels[1]
        self.K_crop[1,2] = self.K[1,2] - (crop_pixels[0])
        self.croped_camera_matrix = np.hstack([np.matmul(self.K_crop,self.R),-np.matmul(self.K_crop,self.t)])
        


    
    def project_on_image(self,points,camera_matrix = True):
        """project 3d points on 2d image

        Args:
            points (np array): 3d points in camera axes
            cam_matrix (np array): camera calibration matrix [K[R|T]]

        Returns:
            pixels (x/u,y/v): _description_
        """
        camera_matrix = self.camera_matrix if camera_matrix else self.croped_camera_matrix
        points_homo = np.hstack((points,np.ones((1,points.shape[0])).T))
        points_3d = np.matmul(camera_matrix,points_homo.T)
        points_2d = (points_3d[:-1, :] / points_3d[-1, :]).T
        return points_2d,points_3d
    
    def rotmat2qvec(self):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = self.R.T.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        self.qvec = np.round(np.array(qvec),7)
        
    

