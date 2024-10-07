import numpy as np
from PIL import Image

class ImageFly():
    def __init__(self,path,frame,cam):
        self.image = np.array(Image.open(f'{path}images/P{frame}CAM{cam + 1}.jpg'))
        y,x = np.where(self.image > 0)
        self.pixels = np.vstack([x,y]).T

    def crop_image(self,delta_xy = 80):
        cm = np.mean(self.pixels,0).astype(int)
        self.croped_image = self.image[cm[0]-delta_xy : cm[0] + delta_xy,cm[1] - delta_xy: cm[1] + delta_xy]


    
    
