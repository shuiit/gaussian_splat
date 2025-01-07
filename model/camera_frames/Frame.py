import cv2
import numpy as np
from PIL import Image
from Camera import Camera
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
class Frame(Camera):
    def __init__(self,path,image_name,idx):
        """
        Initialize a Frame object by loading the corresponding image and voxels and processing pixel data.

        Args:
            path (str): The directory path to the images.
            im_name (str): The name of the image file.
            idx (int): Unique identifier for the image.
        """
      

        self.image_name = f'{image_name}.jpg'
        self.image_id = idx
        self.path = path
        self.frame = int(image_name.split('CAM')[0].split('P')[1])
        super().__init__(self.path,int(image_name.split('CAM')[-1].split('.')[0]) - 1)
        self.load_and_crop_image()
        self.camera_number = idx
    
    def load_image(self):
   
        im = scipy.io.loadmat(f'{self.path}images/{self.image_name.split(".jpg")[0]}.mat')['im']
        bg = np.array((scipy.io.loadmat(f'{self.path}images/bg.mat')['bg']//255).astype(np.uint16))*0+255
        image = Image.fromarray(np.array((im * 255).astype(np.uint8)), mode="L")
        return image,bg
    
    def erode_and_add_bf(self,image,bg,kernel = np.ones((2, 2), np.uint8)):
        
        eroded_image = np.array(cv2.erode(np.array(image), kernel))
        image_with_bg = np.array(bg)
        image_with_bg[eroded_image > 0] = eroded_image[eroded_image > 0]
        return Image.fromarray(image_with_bg),Image.fromarray(eroded_image)

    def load_and_crop_image(self,delta_xy = 80):
        """
        Crop the image around the mean pixel coordinates.

        Args:
            delta_xy (int, optional): The half-width of the cropping area. Default is 80.
        """
        image,bg = self.load_image()
        image,image_no_bg = self.erode_and_add_bf(image,bg)

        pixels = np.vstack(np.where(np.array(image) > 0)).T
        cm = np.mean(pixels,0).astype(int)
        self.bounding_box = [max(0,cm[1] - delta_xy), max(0,cm[0]-delta_xy), max(0,cm[1] - delta_xy) + delta_xy*2 , max(0,cm[0]-delta_xy) + delta_xy*2] # [top left, bottom right]
        self.top_left = [cm[0]-delta_xy,cm[1]-delta_xy]
        self.crop_size = delta_xy*2

        self.camera_calibration_crop(self.top_left) 

        self.image = image.crop(self.bounding_box)
        self.bg  = bg[self.bounding_box[0]:self.bounding_box[2],self.bounding_box[1]:self.bounding_box[2]]
        self.image_no_bg = image_no_bg.crop(self.bounding_box)
        self.image_size = self.image.size
        self.pixels = pixels - self.top_left
        self.cm = np.mean(self.pixels,0)


    def map_3d_2d(self, croped_image = False,use_zbuff = True):
        """
        Map 3D voxel positions to 2D pixel coordinates and store relevant data.

        Args:
            croped_image (bool, optional): Whether to use cropped image pixels. Default is False.
        """
        voxels,pixels_of_voxels = self.z_buffer(croped_camera_matrix = croped_image) if use_zbuff == True else self.map_no_zbuff(croped_camera_matrix = croped_image)
        pixels_from_image = self.croped_pixels if croped_image else self.pixels

        original_projected_pixels = np.vstack((pixels_of_voxels,np.fliplr(pixels_from_image))) # project pixels
        [non_intersect_pixels,cnt] = np.unique(original_projected_pixels,axis = 0,return_counts=True) # identify non intersecting pixels
        non_intersect_pixels = non_intersect_pixels[cnt == 1,:] 

        all_pixels = np.vstack((pixels_of_voxels, non_intersect_pixels)) if use_zbuff == True else pixels_of_voxels
        all_3d_idx = np.full(all_pixels.shape[0], -1)
        all_3d_idx[0:voxels.shape[0]] = voxels[:,3]

        self.pixel_with_idx = np.column_stack((all_pixels, all_3d_idx))
        self.voxels_with_idx = np.column_stack((voxels,np.full(pixels_of_voxels.shape[0],self.image_id),np.arange(pixels_of_voxels.shape[0])))

        # determine the color of every pixel that has a mapping, 
        image_for_color = np.array(self.croped_image) if croped_image == True else np.array(self.image)
        idx = self.pixel_with_idx[:,2] != -1
        pixels = self.pixel_with_idx[idx,0:3].astype(int)
        self.color_of_pixel =  np.array(image_for_color)[pixels[:,1],pixels[:,0]]

    
    def map_no_zbuff(self,croped_camera_matrix = False):
        
        voxels_cam = np.matmul(self.world_to_cam, self.points_in_ew_frame_homo.T).T
        projected = self.project_on_image(self.points_in_ew_frame_homo,croped_camera_matrix)
        pxls = np.round(projected) 
        
        idx_sorted_by_z = voxels_cam[:,2].argsort()
        voxels_sorted_by_z = self.points_in_ew_frame[idx_sorted_by_z,:]
        return voxels_sorted_by_z,pxls[idx_sorted_by_z,:]

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
        """
        Adds homogeneous coordinates to a set of 3D points.

        Args:
            points (np.array): Array of 3D points with shape (n, 3).

        Returns:
            np.array: Array of 3D points in homogeneous coordinates with shape (n, 4).
        """
        return np.column_stack((points,np.ones([points.shape[0],1])))


    def filter_projections_from_bg(self,point3d,croped_image = False):
        """
        Filters 3D points projected onto the image to exclude background pixels.

        Args:
            point3d (np.array): Array of 3D points to be projected.
            croped_image (bool): If True, uses the cropped image without background.
                                If False, uses the full image without background.

        Returns:
            np.array: Boolean array indicating which projected 3D points are on background pixels (True if background).
        """
        image = self.image_no_bg if croped_image == False else self.croped_image_no_bg
        homo_voxels_with_idx = self.add_homo_coords(point3d)
        proj = self.project_on_image(homo_voxels_with_idx,croped_camera_matrix = True)

        idx_round = np.round(np.fliplr(proj)).astype(int)
        idx_to_keep = np.array([True]*idx_round.shape[0])
        pixels_in_image = ((idx_round > 0) & (idx_round <  self.croped_image_size[1])).all(1)
        idx_bg = np.array(image)[idx_round[pixels_in_image,0],idx_round[pixels_in_image,1]] == 0
        idx_to_keep[pixels_in_image] = idx_bg
        return idx_to_keep


    





