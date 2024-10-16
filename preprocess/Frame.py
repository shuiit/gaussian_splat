
import numpy as np
from PIL import Image
from Camera import Camera
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

class Frame(Camera):
    def __init__(self,path,im_name,point_3d,real_coord,idx):
        """
        Initialize a Frame object by loading the corresponding image and voxels and processing pixel data.

        Args:
            path (str): The directory path to the images.
            im_name (str): The name of the image file.
            point_3d (DataFrame): A DataFrame containing 3D points.
            real_coord (numpy.ndarray): Array of real coordinates associated with the points.
            idx (int): Unique identifier for the image.
        """
        self.image = Image.open(f'{path}images/{im_name}')
        self.image_id = idx
        y,x = np.where(np.array(self.image) > 0)
        self.pixels = np.vstack([y,x]).T
        self.path = path
        self.frame = int(im_name.split('CAM')[0].split('P')[1])
        self.real_coord_frame = real_coord[real_coord[:,3] == self.frame,:]
        self.points_in_idx = point_3d
        super().__init__(self.path,int(im_name.split('CAM')[-1].split('.')[0]) - 1)
        self.crop_image() 
        self.idx_to_real()
        self.camera_number = idx

    
    def crop_image(self,delta_xy = 80):
        """
        Crop the image around the mean pixel coordinates.

        Args:
            delta_xy (int, optional): The half-width of the cropping area. Default is 80.
        """
        cm = np.mean(self.pixels,0).astype(int)
        im_to_crop = self.image.copy()
        bounding_box = [max(0,cm[1] - delta_xy), max(0,cm[0]-delta_xy), max(0,cm[1] - delta_xy) + delta_xy*2 , max(0,cm[0]-delta_xy) + delta_xy*2] # [top left, bottom right]
        self.croped_image = im_to_crop.crop(bounding_box)
        self.top_left = [cm[0]-delta_xy,cm[1]-delta_xy]
        self.crop_size = delta_xy*2
        self.croped_pixels = self.pixels - self.top_left
        self.camera_calibration_crop(self.top_left) 


    def map_3d_2d(self, croped_image = False):
        """
        Map 3D voxel positions to 2D pixel coordinates and store relevant data.

        Args:
            croped_image (bool, optional): Whether to use cropped image pixels. Default is False.
        """
        voxels,pixels_of_voxels = self.z_buffer(croped_camera_matrix = croped_image)
        pixels_from_image = self.croped_pixels if croped_image else self.pixels

        original_projected_pixels = np.vstack((pixels_of_voxels,np.fliplr(pixels_from_image))) # project pixels
        [non_intersect_pixels,cnt] = np.unique(original_projected_pixels,axis = 0,return_counts=True) # identify non intersecting pixels
        non_intersect_pixels = non_intersect_pixels[cnt == 1,:] 

        all_pixels = np.vstack((pixels_of_voxels, non_intersect_pixels))
        all_3d_idx = np.full(all_pixels.shape[0], -1)
        all_3d_idx[0:voxels.shape[0]] = voxels[:,3]

        self.pixel_with_idx = np.column_stack((all_pixels, all_3d_idx))
        self.voxels_with_idx = np.column_stack((voxels,np.full(pixels_of_voxels.shape[0],self.image_id),np.arange(pixels_of_voxels.shape[0])))

        # determine the color of every pixel that has a mapping, 
        image_for_color = np.array(self.croped_image) if croped_image == True else np.array(self.image)
        idx = self.pixel_with_idx[:,2] != -1
        pixels = self.pixel_with_idx[idx,0:3].astype(int)
        self.color_of_pixel =  np.array(image_for_color)[pixels[:,1],pixels[:,0]]

    def idx_to_real(self):
        """
        Convert indexed points to real-world coordinates in 3D space.
        """
        self.points_in_ew_frame = np.array([self.real_coord_frame[self.points_in_idx[ax] - 1,idx] for idx,ax in enumerate(['X','Y','Z'])]).T
        self.points_in_ew_frame_homo = np.hstack((self.points_in_ew_frame,np.ones([self.points_in_ew_frame.shape[0],1])))
        self.points_in_ew_frame  = np.column_stack((self.points_in_ew_frame,np.arange(1,self.points_in_ew_frame.shape[0] + 1)))

    def z_buffer(self,croped_camera_matrix = False):
        """
        Compute the z-buffer for 3D points, projecting them onto the 2D image plane.

        Args:
            croped_camera_matrix (bool, optional): Whether to use a cropped camera matrix. Default is False.

        Returns:
            tuple: A tuple containing sorted voxel positions and pixel coordinates.
        """
        voxels_cam = np.matmul(self.world_to_cam, self.points_in_ew_frame_homo.T).T
        projected = self.project_on_image(self.points_in_ew_frame_homo,croped_camera_matrix)
        pxls = np.round(projected) 
        idx_sorted_by_z = voxels_cam[:,2].argsort()
        voxels_sorted_by_z = self.points_in_ew_frame[idx_sorted_by_z,:]
        [pixels,idx] = np.unique(pxls[idx_sorted_by_z,:], axis=0,return_index=True)
        return voxels_sorted_by_z[idx,:],pixels
    
    def add_homo_coords(self,points):
        return np.column_stack((points,np.ones([points.shape[0],1])))



    





