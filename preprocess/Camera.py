
import numpy as np
import scipy.io
import math 

class Camera():
    def __init__(self,path,camera_number, cam = False):
        self.path = path
        cam,get_cam_mat = (scipy.io.loadmat(f'{path}/camera_KRX0.mat'),camera_number) if cam == False else (cam,0)
        self.K = cam['camera'][0:3,0:3,get_cam_mat]
        self.R = cam['camera'][0:3,3:6,get_cam_mat]
        self.X0 = cam['camera'][0:3,6:7,get_cam_mat]
        self.t = -np.matmul(self.R,self.X0)
        self.camera_number = camera_number + 1
        self.K[1,2]  = self.K[1,2] - 1  
        self.K[0,2]  = self.K[0,2] - 1
        self.fx = self.K[0,0]
        self.fy = self.K[1,1]
        self.cx = self.K[0,2]
        self.cy = self.K[1,2]
        self.znear = 0.000000001
        self.zfar = 100
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
    
    def focal2fov(self,focal, pixels):
        return 2*math.atan(pixels/(2*focal))
    

    
    def getProjectionMatrix(self,im_size):

        fovy = self.focal2fov(self.fy, im_size[1])
        fovx = self.focal2fov(self.fx, im_size[0])
        tanHalfFovY = math.tan((fovy / 2))
        tanHalfFovX = math.tan((fovx / 2))

        top = tanHalfFovY * self.znear
        bottom = -top
        right = tanHalfFovX * self.znear
        left = -right

        P = np.zeros((4, 4))

        z_sign = 1.0

        P[0, 0] = 2.0 * self.znear / (right - left)
        P[1, 1] = 2.0 * self.znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        P[0, 2] = (right + left) / (right - left) + (2 * self.cx / im_size[0]) - 1
        P[1, 2] = (top + bottom) / (top - bottom) + (2 * self.cy / im_size[1]) - 1
        self.projection = P
        world_view_transform = np.vstack((self.world_to_cam,[0,0,0,1]))
        self.full_proj_transform = np.matmul(P,world_view_transform)  # Shape: (1, N, K)

    
    


        
    

