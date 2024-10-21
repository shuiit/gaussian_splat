import os
import numpy as np



class GenerateGsInput():
    def __init__(self,path,frames):
        self.path = path
        self.sparse_dir = f'{self.path}/input_data_for_gs/sparse/'
        self.image_dir = f'{self.path}/input_data_for_gs/images/'
        self.num_cam = len(frames)
        self.frames = frames

        if not os.path.exists(self.sparse_dir):
            os.makedirs(self.sparse_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def save_croped_images(self,croped_image = False):
        if croped_image == True:
            image_rgb = [self.frames[image].croped_image.convert('RGB') for image in self.frames.keys()]
        else:
            image_rgb = [self.frames[image].image.convert('RGB') for image in self.frames.keys()]

        for (im_name,image) in zip(self.frames.keys(),image_rgb):
            image.save(f'{self.path}/input_data_for_gs/images/{im_name}', format='JPEG', subsampling=0, quality=100)


    def generate_camera_text(self,croped_image = False):

        with open(f'{self.sparse_dir}/cameras.txt', "w") as file:
            file.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            file.write(f"# Number of cameras: {self.num_cam}\n")
            for image in self.frames.values(): 
                intrinsic = image.K_crop if croped_image == True else image.K
                intrinsic = np.round(np.array(intrinsic),2)
                image_size = list(self.frames.values())[0].croped_image.size if croped_image == True else list(self.frames.values())[0].image.size
                camera_data = [image.camera_number, "PINHOLE", image_size[0], image_size[1], intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]]
                file.write(" ".join(map(str, camera_data)) + "\n")


    def generate_image_text(self):

        with open(f'{self.sparse_dir}/images.txt', "w") as file:
            mean_obs = sum([sum(self.frames[frames_name].pixel_with_idx[:,2] != -1) for frames_name in self.frames.keys()])/len(self.frames)
            file.write("# Image list with two lines of data per image:\n#")
            file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            file.write(f"# Number of images: {len(self.frames)}, mean observations per image: {mean_obs} \n")

            for idx,(frames_name,image) in enumerate(self.frames.items()): 

                camera_data = [idx] + list(image.qvec) + list(image.t.T[0]) + [image.camera_number] + [frames_name]
                file.write(" ".join(map(str, camera_data)) + "\n")
                size_pixels = self.frames[frames_name].pixel_with_idx.shape
                file.write(" ".join(
                    f"{int(value)}" if (i % 3 == 2) else f"{value:.3f}"  # Format every third value without decimals
                    for i, value in enumerate(self.frames[frames_name].pixel_with_idx.reshape(size_pixels[0] * size_pixels[1]))
                ) + "\n")

    def generate_points3d_text(self,voxel_dict,colors_dict):
        with open(f'{self.sparse_dir}/points3D.txt', "w") as file:
            file.write("# 3D point list with one line of data per point:\n#")
            file.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            file.write(f"#   Number of points: {len(voxel_dict)}, mean track length: {np.mean([len(value) for value in voxel_dict.values()])}\n")

            for key in voxel_dict.keys(): 
                data_3d = [key] + voxel_dict[key][0:3] + colors_dict[key] + [0.1] +  voxel_dict[key][3:] 
                file.write(" ".join(map(str, data_3d)) + "\n")