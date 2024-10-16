
import numpy as np
import scipy.io

class Camera():
    def __init__(self,path,camera_number):
        self.path = path
        cam = scipy.io.loadmat(f'{path}/camera_KRX0.mat')
        self.K = cam['camera'][0:3,0:3,camera_number]
        self.R = cam['camera'][0:3,3:6,camera_number]
        self.X0 = cam['camera'][0:3,6:7,camera_number]
        self.t = -np.matmul(self.R,self.X0)
        self.camera_number = camera_number + 1
        self.K[1,2]  = self.K[1,2] - 1  
        self.K[0,2]  = self.K[0,2] - 1
        self.world_to_cam = np.hstack([self.R,self.t])
        self.camera_matrix = np.hstack([np.matmul(self.K,self.R),np.matmul(self.K,self.t)])
        self.rotmat2qvec()
        


    def camera_calibration_crop(self,crop_pixels):
        """updates the intrinsic K matrix for croped images

        Args:
            crop_pixels (np array): loaction of top left pixel 
        """
        self.K_crop = self.K.copy()
        self.K_crop[0,2] = self.K[0,2] - crop_pixels[1]
        self.K_crop[1,2] = self.K[1,2] - (crop_pixels[0])
        self.croped_camera_matrix = np.hstack([np.matmul(self.K_crop,self.R),np.matmul(self.K_crop,self.t)])
        


    
    def project_on_image(self,points,croped_camera_matrix = False):
        """project 3d points on 2d image

        Args:
            points (np array): 3d points in camera axes
            cam_matrix (np array): camera calibration matrix [K[R|T]]

        Returns:
            pixels (x/u,y/v): pixels in image plane
        """
        camera_matrix = self.croped_camera_matrix if croped_camera_matrix else self.camera_matrix
        points_2d = np.matmul(camera_matrix,points.T)
        points_2d = (points_2d[:-1, :] / points_2d[-1, :]).T
        return points_2d
    

    def rotate_world_to_cam(self,points):
        """Rotate points from world coordinates to camera coordinates.

        Args:
        points (ndarray): Array of points in world coordinates (shape: [n, 3]).

        Returns:
            ndarray: Array of points in camera coordinates (shape: [n, 3]).
        """
        return np.matmul(self.world_to_cam , points).T
    
    
    def rotmat2qvec(self):
        """Convert a rotation matrix to a quaternion vector
        Taken from colmap loader (gaussian-splatting)-- probably taken from colmap 
        """
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = self.R.flat
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
        
    

