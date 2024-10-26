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
        im = scipy.io.loadmat(f'{path}images/{im_name}.mat')['im']
        bg = scipy.io.loadmat(f'{path}images/bg.mat')['bg']
        self.bg = np.array((bg//255).astype(np.uint16))*0+255
        self.image = Image.fromarray(np.array((im * 255).astype(np.uint8)), mode="L")
        y,x = np.where(np.array(self.image) > 0)
        kernel = np.ones((2, 2), np.uint8) 
        eroded_image = np.array(cv2.erode(np.array(self.image), kernel))
        self.image_with_bg = np.array(self.bg.copy())
        self.image_with_bg[eroded_image > 0] = eroded_image[eroded_image > 0]
        self.image_no_bg =  Image.fromarray(eroded_image)
        self.image =  Image.fromarray(self.image_with_bg)
        # self.image =np.array(self.image)
        # self.image[self.image == 0] = 255

        # self.image = 256 - np.array(self.image)
        # self.image[self.image == 256] = 0

        # diff_image_bg = np.array(self.image).astype(np.uint8) - np.array(self.bg).astype(np.uint8)
        # diff_image_bg = diff_image_bg*(np.array(self.image_no_bg) > 0)

        # alpha = (np.array(self.bg).astype(np.float)/255 - np.array(self.image_with_bg).astype(np.float)/255)/(1 -np.array(self.bg).astype(np.float)/255 )

        # I_observed = np.array(self.image_with_bg).astype(np.float)
        # I_background =np.array(self.bg).astype(np.float)
        # # Calculate alpha mask based on the absolute difference
        # alpha = np.abs(I_observed - I_background) / 255.0
        # alpha = np.clip(alpha, 0, 1)  # Ensure alpha is within [0, 1]

        # rgb_im = np.dstack([I_observed]*3)
        # I_new_background = np.dstack([I_observed*0+255,I_observed*0,I_observed*0]*3)

        # # Blend the observed image with the new background using the alpha mask
        # I_new_observed = alpha * I_observed 

        # # Ensure the result is in valid range and type
        # I_new_observed = np.clip(I_new_observed, 0, 255).astype(np.uint8)


        # I_new_background = np.dstack([I_new_observed/255.0,I_new_observed*0,I_new_observed*0])



        self.image_id = idx
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
        im_to_crop_nobg = self.image_no_bg.copy()
        bounding_box = [max(0,cm[1] - delta_xy), max(0,cm[0]-delta_xy), max(0,cm[1] - delta_xy) + delta_xy*2 , max(0,cm[0]-delta_xy) + delta_xy*2] # [top left, bottom right]
        self.croped_image = im_to_crop.crop(bounding_box)
        self.croped_image_no_bg = im_to_crop_nobg.crop(bounding_box)
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


    def match_hist(self,ref_image):
        self.croped_image = match_histograms(np.array(self.croped_image), np.array(ref_image))

    





