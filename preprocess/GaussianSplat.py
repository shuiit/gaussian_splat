
import numpy as np
import scipy.io
from plyfile import PlyData, PlyElement
import Plotters

class GaussianSplat():
    def __init__(self,path = None,vertices = None):
        self.path = path
        self.vertices = PlyData.read(path)["vertex"] if vertices is None else vertices
        self.xyz = np.column_stack((self.vertices["x"], self.vertices["y"], self.vertices["z"]))
        self.scale = np.exp(np.column_stack(([self.vertices["scale_0"], self.vertices["scale_1"], self.vertices["scale_2"]])))

        SH_C0 = 0.28209479177387814
        self.color = np.column_stack([
            0.5 + SH_C0 * self.vertices["f_dc_0"],
            0.5 + SH_C0 * self.vertices["f_dc_1"],
            0.5 + SH_C0 * self.vertices["f_dc_2"],
            1 / (1 + np.exp(-self.vertices["opacity"])),])
        self.rot = np.column_stack([self.vertices["rot_0"], self.vertices["rot_1"], self.vertices["rot_2"], self.vertices["rot_3"]])

    def filter(self,filter_by,**kwargs):
        return GaussianSplat(vertices = self.vertices[filter_by],**kwargs)

    def save_gs(self,name = '_filtered'):
        filtered_element = PlyElement.describe(self.vertices, 'vertex')
        PlyData([filtered_element]).write(f'{self.path.split(".ply")[0]}{name}.ply')

        
