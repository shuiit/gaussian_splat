
import numpy as np
import scipy.io
from plyfile import PlyData, PlyElement
import Plotters
import sh_utils

class GaussianSplat():
    def __init__(self,path = None,vertices = None,block_xy = [16,16], image_size = [160,160],sh = None, deg_sh = 0):
        self.path = path
        self.vertices = PlyData.read(path)["vertex"] if vertices is None else vertices
        self.xyz = np.column_stack((self.vertices["x"], self.vertices["y"], self.vertices["z"]))
        self.scale = np.exp(np.column_stack(([self.vertices["scale_0"], self.vertices["scale_1"], self.vertices["scale_2"]])))
        
        self.sh = np.column_stack([self.vertices[key] for key in self.vertices.data.dtype.names if 'rest' in key or 'dc' in key]) if sh is None else sh
        self.image_size = image_size
        self.block_xy = block_xy
        self.grid = [int((self.image_size[0] + self.block_xy[0] - 1)/self.block_xy[0]),int((self.image_size[1] + self.block_xy[1] - 1)/self.block_xy[1])]
        self.opacity = 1 / (1 + np.exp(-self.vertices["opacity"]))#self.vertices["opacity"]         
        self.rot = np.column_stack([self.vertices["rot_0"], self.vertices["rot_1"], self.vertices["rot_2"], self.vertices["rot_3"]])
        self.get_color(0)
        

    def projection_filter(self,frames,point3d,**kwargs):
        return np.column_stack([image.filter_projections_from_bg(point3d,**kwargs) for image in frames.values()]).any(axis = 1) == False


    def filter(self,filter_by,**kwargs):
        return GaussianSplat(vertices = self.vertices[filter_by], sh = self.sh[filter_by,:],**kwargs)

    def save_gs(self,name = '_filtered'):
        filtered_element = PlyElement.describe(self.vertices, 'vertex')
        PlyData([filtered_element]).write(f'{self.path.split(".ply")[0]}{name}.ply')

    def q_array_to_rotmat(self,q):
        return np.column_stack(
            [1.0 - 2.0 * (q[:,2] * q[:,2] + q[:,3] * q[:,3]), 2.0 * (q[:,1]  * q[:,2] - q[:,0] * q[:,3] ), 2.0 * (q[:,1] * q[:,3]  + q[:,0] * q[:,2]),
            2.0 * (q[:,1] * q[:,2] + q[:,0] * q[:,3] ), 1.0 - 2.0 * (q[:,1] * q[:,1] + q[:,3]  * q[:,3] ), 2.0 * (q[:,2] * q[:,3]  - q[:,0] * q[:,1]),
            2.0 * (q[:,1] * q[:,3]  - q[:,0] * q[:,2]), 2.0 * (q[:,2] * q[:,3]  + q[:,0] * q[:,1]), 1.0 - 2.0 * (q[:,1] * q[:,1] + q[:,2] * q[:,2])])


    def calc_cov3d(self):
        scale = np.eye(3) * self.scale[:,np.newaxis,:] # size of the gaussians along X,Y,Z
        q = (self.rot.T/ np.linalg.norm(self.rot,axis =1)).T # orientation of the gaussian (quaternion)
        rot_mat = self.q_array_to_rotmat(q) # converts the array of quaternions into rotation matrices
        rot_mat = rot_mat.reshape(rot_mat.shape[0],3,3) 
        scale_rot = rot_mat @ scale   # scales the rotation matrix -> generates a gaussian in the orientation of q. 
        self.cov3d = scale_rot @ scale_rot.transpose(0,2,1) # the covariance matrix 

    def calc_cov2d(self,camera, image_size = [160,160]):
        # camera parameters:
        fxfy = [camera.K[0,0],camera.K[1,1]]
        tan_fov = np.array([camera.focal2fov(focal, size) for focal,size in zip(fxfy,image_size)])
        viewmat = camera.world_to_cam

        # project 3d coordinates and clip to keep only pixels in screen space
        projected = np.matmul(viewmat , np.column_stack((self.xyz,np.ones(self.xyz.shape[0]))).T).T
        limxy = np.tile(tan_fov*1.3,(projected.shape[0],1))
        projected[:, :2] = np.minimum(limxy, np.maximum(-limxy, projected[:,0:2]/projected[:,2:])) *projected[:,2:]

        # calculate the Jacobian - the change in reprojection. they use the Jacobian to define the projection. its a Taylor expansion.
        jacobian = self.calc_jacobian(fxfy[0],fxfy[1],projected)
        jacobian = jacobian.reshape((jacobian.shape[0],3,3))

        # rotate the Jacobian to the correct frame of reference
        tile_viwe_mat = np.tile(viewmat[0:3,0:3].T,(jacobian.shape[0],1,1))
        T = tile_viwe_mat @ jacobian
        cov = T.transpose(0,2,1) @ self.cov3d.transpose(0,2,1) @ T
        self.cov2d_matrix = cov
        self.cov2d = np.squeeze(np.dstack((cov[:,0,0],cov[:,0,1],cov[:,1,1])))

        self.cov2d[:,0] = self.cov2d[:,0]+ 0.3
        self.cov2d[:,2] = self.cov2d[:,2]+ 0.3

        self.det = self.cov2d[:,0]*self.cov2d[:,2] - self.cov2d[:,1]*self.cov2d[:,1]
        self.inv_det = 1/self.det
        self.conic = np.column_stack((self.cov2d[:,2]*self.inv_det,-self.cov2d[:,1]*self.inv_det,self.cov2d[:,0]*self.inv_det,self.opacity))

        
        self.radius = self.compute_radius()
        self.projected = projected

    def compute_radius(self):
        mid = 0.5*(self.cov2d[:,0] + self.cov2d[:,2])
        lambda1 = mid + np.sqrt(np.maximum(0.1,mid * mid - self.det))
        lambda2 = mid - np.sqrt(np.maximum(0.1,mid * mid - self.det))
        return np.ceil(3.0 * np.sqrt(np.maximum(lambda1, lambda2)))

    def calc_jacobian(self,fx,fy,projected):

        zero_np = np.zeros((projected.shape[0],1))
        return np.column_stack((
            fx / projected[:,2:], zero_np, - (fx * projected[:,0:1]) / (projected[:,2:] ** 2),
            zero_np, fy / projected[:,2:], - (fy * projected[:,1:2]) / (projected[:,2:] ** 2),
            zero_np, zero_np, zero_np
        ))

    def get_rect(self,cam):   
        xyz_homo = self.add_homo_coords(self.xyz)
        pixel = cam.project_on_image(xyz_homo)
        xy_up_left_corner = np.minimum(self.grid,(np.maximum(0,pixel - self.radius[:,np.newaxis]) / self.block_xy).astype(int))
        xy_bot_right_corner = np.minimum(self.grid,((np.maximum(0,(pixel + self.radius[:,np.newaxis] + self.block_xy - 1) / self.block_xy)))).astype(int)
        return xy_up_left_corner,xy_bot_right_corner

    def add_homo_coords(self,points):
        return np.column_stack((points,np.ones([points.shape[0],1])))
    
    def get_color(self,deg, **kwargs):
        self.color = sh_utils.rgb_from_sh(deg,self.sh, **kwargs)
    




