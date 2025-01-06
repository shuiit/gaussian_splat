
import os.path
import open3d as o3d
import numpy as np
import Plotter 


class Skin():
    def __init__(self, parts,path_to_mesh, scale = 1):
        self.parts = {part : idx for idx,part in enumerate(parts)}
        self.scale = scale
        self.load_skin(path_to_mesh)
        

    def load_skin(self,path_to_mesh):
        ptcloud_parts = {part : self.load_mesh(f'{path_to_mesh}/{part}.stl',idx) for part,idx in self.parts.items() if os.path.isfile(f'{path_to_mesh}/{part}.stl')}
        skin = np.vstack(list(ptcloud_parts.values()))
        self.ptcloud_skin = skin[:,0:3]*self.scale
        self.ptcloud_part_idx = skin[:,3]

    def load_mesh(self,path_to_mesh,idx_mesh):
        mesh = o3d.io.read_triangle_mesh(path_to_mesh)
        pt_cloud = np.asarray(mesh.vertices)
        idx_of_part = np.ones((pt_cloud.shape[0],1))*idx_mesh
        return np.hstack((pt_cloud,idx_of_part))
    
    def translate_ptcloud_skin(self,translation):
        self.ptcloud_skin = self.ptcloud_skin - translation


    def calculate_weights(self,skeleton,constant_weight = [False,'right_wing_root','left_wing_root'],**kwargs):
        self.weight = np.vstack([skeleton.calculate_weight(self.get_part(part), constant_weight = constant_weight,**kwargs) for constant_weight,part in zip(constant_weight,self.parts.keys())])





    def plot_skin(self,fig,legend,skip = 10,**kwargs):
        Plotter.scatter3d(fig,self.ptcloud_skin[::skip,:],legend,**kwargs)
    



    def get_part(self,part):
        return self.ptcloud_skin[self.ptcloud_part_idx == self.parts[part],:]


    # def calculate_weight(self,skeleton, idx):
    #     skeleton.calculate_weight(body)

    


    
