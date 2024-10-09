import os
import numpy as np



class GenerateGsInput():
    def __init__(self,path,images,cameras):
        self.path = path
        self.sparse_dir = f'{self.path}/input_data_for_gs/sparse/'
        self.image_dir = f'{self.path}/input_data_for_gs/images/'
        self.num_cam = len(cameras)
        self.images = images
        self.cameras = cameras

        if not os.path.exists(self.sparse_dir):
            os.makedirs(self.sparse_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def save_croped_images(self):
        [self.images[image].croped_image.save(f'{self.path}/input_data_for_gs/images/{image}') for image in self.images.keys()]




    def generate_camera_text(self):

        with open(f'{self.sparse_dir}/cameras.txt', "w") as file:
            file.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            file.write(f"# Number of cameras: {self.num_cam}\n")
            for camera in self.cameras.values(): 
                intrinsic = np.round(np.array(camera.K),2)
                image_size = list(self.images.values())[0].crop_size
                camera_data = [camera.camera_number, "PINHOLE", image_size, image_size, intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]]
                file.write(" ".join(map(str, camera_data)) + "\n")


