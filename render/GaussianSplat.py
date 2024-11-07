
import numpy as np
import scipy.io
from plyfile import PlyData, PlyElement
import Plotters
import sh_utils

class GaussianSplat():
    """
        Initializes the GaussianSplat class with vertices, camera parameters, and Gaussian parameters.
        
        Args:
            path (str): Path to the .ply file containing vertex data.
            vertices (np.array): Array of vertices. Loaded from file if not provided.
            block_xy (list): Block size for 2D grid representation.
            image_size (list): Size of the image in pixels.
            sh (np.array): Spherical harmonics coefficients. Loaded from file if not provided.
        """
    def __init__(self,path = None,vertices = None,block_xy = [16,16], image_size = [160,160],sh = None):
        self.path = path
        self.vertices = PlyData.read(path)["vertex"] if vertices is None else vertices
        self.xyz = np.column_stack((self.vertices["x"], self.vertices["y"], self.vertices["z"]))
        self.scale = np.exp(np.column_stack(([self.vertices["scale_0"], self.vertices["scale_1"], self.vertices["scale_2"]])))
        self.opacity = 1 / (1 + np.exp(-self.vertices["opacity"]))#self.vertices["opacity"]         
        self.rot = np.column_stack([self.vertices["rot_0"], self.vertices["rot_1"], self.vertices["rot_2"], self.vertices["rot_3"]])


        self.sh = np.column_stack([self.vertices[key] for key in self.vertices.data.dtype.names if 'rest' in key or 'dc' in key]) if sh is None else sh
        self.image_size = image_size
        self.block_xy = block_xy
        self.grid = [int((self.image_size[0] + self.block_xy[0] - 1)/self.block_xy[0]),int((self.image_size[1] + self.block_xy[1] - 1)/self.block_xy[1])]
        self.get_color(0)
        

    def rearange_gs(self,idx_to_rearange):
        """
        Rearranges the Gaussian splats based on the given indices.
        
        Args:
            idx_to_rearange (np.array): Array of indices to rearrange vertices and associated properties.
        """
        self.vertices = self.vertices[idx_to_rearange]
        self.xyz = self.xyz[idx_to_rearange]
        self.scale = self.scale[idx_to_rearange]
        self.opacity = self.opacity[idx_to_rearange]
        self.rot = self.rot[idx_to_rearange]
        self.color = self.color[idx_to_rearange]
        self.sh = self.sh[idx_to_rearange]


    def projection_filter(self,frames,point3d,**kwargs):
        """
        Filters projected 3D points to exclude background pixels across multiple frames.
        
        Args:
            frames (dict): Dictionary of frame objects for projection filtering.
            point3d (np.array): 3D points to project.
        
        Returns:
            np.array: Boolean array indicating which points are not in the background for any frame.
        """
        return np.column_stack([image.filter_projections_from_bg(point3d,**kwargs) for image in frames.values()]).any(axis = 1) == False


    def filter(self,filter_by,**kwargs):
        """
        Creates a new GaussianSplat object filtered by a boolean array.
        
        Args:
            filter_by (np.array): Boolean array to filter vertices.
        
        Returns:
            GaussianSplat: New GaussianSplat instance with filtered vertices.
        """
        return GaussianSplat(vertices = self.vertices[filter_by], sh = self.sh[filter_by,:],**kwargs)

    def save_gs(self,name = '_filtered'):
        """
        Saves the filtered vertices to a new .ply file.
        
        Args:
            name (str): Suffix for the output filename.
        """
        filtered_element = PlyElement.describe(self.vertices, 'vertex')
        PlyData([filtered_element]).write(f'{self.path.split(".ply")[0]}{name}.ply')

    def q_array_to_rotmat(self,q):
        """
        Converts quaternion array to rotation matrix.
        
        Args:
            q (np.array): Array of quaternions.
        
        Returns:
            np.array: Corresponding rotation matrix.
        """
        return np.column_stack(
            [1.0 - 2.0 * (q[:,2] * q[:,2] + q[:,3] * q[:,3]), 2.0 * (q[:,1]  * q[:,2] - q[:,0] * q[:,3] ), 2.0 * (q[:,1] * q[:,3]  + q[:,0] * q[:,2]),
            2.0 * (q[:,1] * q[:,2] + q[:,0] * q[:,3] ), 1.0 - 2.0 * (q[:,1] * q[:,1] + q[:,3]  * q[:,3] ), 2.0 * (q[:,2] * q[:,3]  - q[:,0] * q[:,1]),
            2.0 * (q[:,1] * q[:,3]  - q[:,0] * q[:,2]), 2.0 * (q[:,2] * q[:,3]  + q[:,0] * q[:,1]), 1.0 - 2.0 * (q[:,1] * q[:,1] + q[:,2] * q[:,2])])


    def calc_cov3d(self):
        """
        Calculates 3D covariance matrices based on scale and rotation (quaternion) for each Gaussian.
        """
        scale = np.eye(3) * self.scale[:,np.newaxis,:] # size of the gaussians along X,Y,Z
        q = (self.rot.T/ np.linalg.norm(self.rot,axis =1)).T # orientation of the gaussian (quaternion)
        rot_mat = self.q_array_to_rotmat(q) # converts the array of quaternions into rotation matrices
        rot_mat = rot_mat.reshape(rot_mat.shape[0],3,3) 
        scale_rot = rot_mat @ scale   # scales the rotation matrix -> generates a gaussian in the orientation of q. 
        self.cov3d = scale_rot @ scale_rot.transpose(0,2,1) # the covariance matrix 

    def calc_cov2d(self,camera, image_size = [160,160]):
        """
        Calculates 2D covariance matrices for each Gaussian projected onto an image plane.
        
        Args:
            camera: Camera object with intrinsic and extrinsic parameters.
            image_size (list): Size of the output image.
        """
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

        # The original covariance matrix : V
        # we multipy by M, the projection matrix. we get: MVM^T 
        # this transdorms from object to camera, then apply the V (covariance) transformation, scaling and shearing the data
        # then multiply by M again, going back to camera FoR
        # Jacobian: used as a taylor expansion, to generate alocal linear transformation around a pixel. the Jacobian is trated as anew transformation 
        # and it is multiplied by the covariance matrix. 
        tile_viwe_mat = np.tile(viewmat[0:3,0:3].T,(jacobian.shape[0],1,1))
        T = tile_viwe_mat @ jacobian
        cov = T.transpose(0,2,1) @ self.cov3d.transpose(0,2,1) @ T
        # cov = jacobian @ tile_viwe_mat @ self.cov3d.transpose(0,2,1) @ tile_viwe_mat.transpose(0,2,1) @  jacobian.transpose(0,2,1)
        self.cov2d_matrix = cov
        self.cov2d = np.squeeze(np.dstack((cov[:,0,0],cov[:,0,1],cov[:,1,1])))

        # Adjust covariance and compute conic parameters
        # We use conic parameters to plot the gaussian on 2d. 
        # conic: Ax^2 + Bxy + Cy^2 = sigma_y^2*x^2 + sigma_xy*xy + sigma_x^2*y^2
        # we devide by the area to scale the reprojection. (the covariance will be the same if we have mm or m, we want to be in the right scaling)
        self.cov2d[:,0] = self.cov2d[:,0]+ 0.3
        self.cov2d[:,2] = self.cov2d[:,2]+ 0.3

        self.det = self.cov2d[:,0]*self.cov2d[:,2] - self.cov2d[:,1]*self.cov2d[:,1]
        self.inv_det = 1/self.det
        self.conic = np.column_stack((self.cov2d[:,2]*self.inv_det,-self.cov2d[:,1]*self.inv_det,self.cov2d[:,0]*self.inv_det,self.opacity))

        
        self.radius = self.compute_radius()
        self.projected = projected

    def compute_radius(self):
        """
        Computes the radius for each Gaussian splat based on covariance values.
        
        Returns:
            np.array: Radius values for each Gaussian splat.
        """
        mid = 0.5*(self.cov2d[:,0] + self.cov2d[:,2])
        lambda1 = mid + np.sqrt(np.maximum(0.1,mid * mid - self.det))
        lambda2 = mid - np.sqrt(np.maximum(0.1,mid * mid - self.det))
        return np.ceil(3.0 * np.sqrt(np.maximum(lambda1, lambda2)))


    def calc_jacobian(self,fx,fy,projected):
        """
        Calculates the Jacobian for each Gaussian splat based on projected points and camera parameters.
        
        Args:
            fx (float): Focal length along x-axis.
            fy (float): Focal length along y-axis.
            projected (np.array): Projected 3D points.
        
        Returns:
            np.array: Jacobian matrices for each projected point.
        """
        zero_np = np.zeros((projected.shape[0],1))
        return np.column_stack((
            fx / projected[:,2:], zero_np, - (fx * projected[:,0:1]) / (projected[:,2:] ** 2),
            zero_np, fy / projected[:,2:], - (fy * projected[:,1:2]) / (projected[:,2:] ** 2),
            zero_np, zero_np, zero_np
        ))


    def get_rect(self,cam):   
        """
        Calculates the upper-left and bottom-right corners of the bounding boxes for each projected point.

        Args:
            cam: Camera object used for projecting 3D points onto the image plane.
            
        Returns:
            Tuple of np.ndarray: 
                - Upper-left corner coordinates (xy_up_left_corner) of bounding boxes.
                - Bottom-right corner coordinates (xy_bot_right_corner) of bounding boxes.
        """
        pixel = cam.project_with_proj_mat(self.xyz)[:,0:2]
        xy_up_left_corner = np.minimum(self.grid,(np.maximum(0,pixel - self.radius[:,np.newaxis]) / self.block_xy).astype(int))
        xy_bot_right_corner = np.minimum(self.grid,((np.maximum(0,(pixel + self.radius[:,np.newaxis] + self.block_xy - 1) / self.block_xy)))).astype(int)
        return xy_up_left_corner,xy_bot_right_corner


    def get_color(self,deg, **kwargs):
        """
        Computes RGB color values from spherical harmonics coefficients for each vertex.

        Args:
            deg (int): Degree of spherical harmonics to consider for color computation.
            **kwargs: Additional keyword arguments for `sh_utils.rgb_from_sh`.
            
        Sets:
            self.color (np.array): Array of RGB color values computed for each vertex.
        """
        self.color = sh_utils.rgb_from_sh(deg,self.sh, **kwargs)





