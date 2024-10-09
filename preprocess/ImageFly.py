import numpy as np
from PIL import Image
from Camera import Camera

class ImageFly(Camera):
    def __init__(self,path,im_name):
        self.image = Image.open(f'{path}images/{im_name}')
        y,x = np.where(np.array(self.image) > 0)
        self.pixels = np.vstack([y,x]).T
        self.path = path
        self.camera_number = int(im_name.split('CAM')[-1].split('.')[0])
        self.frame = int(im_name.split('CAM')[0].split('P')[1])
        super().__init__(self.path,self.camera_number - 1)
        self.crop_image()

    def crop_image(self,delta_xy = 80):
        cm = np.mean(self.pixels,0).astype(int)
        im_to_crop = self.image.copy()
        self.croped_image = im_to_crop.crop([cm[1] - delta_xy, cm[0]-delta_xy, cm[1] + delta_xy , cm[0] + delta_xy])
        self.top_left = [cm[0]-delta_xy,cm[1]-delta_xy]
        self.crop_size = delta_xy*2
        self.croped_pixels = self.pixels - self.top_left
        self.camera_calibration_crop(self.top_left) 

    
    
