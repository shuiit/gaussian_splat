import numpy as np
class Render():
    def __init__(self,gs,cam,tiles = [1,10],block_xy = [16,16], image_size = [160,160],sh = None):
        
        pixels,points_camera = cam.rotate_and_project_with_proj_mat(gs.xyz, 160)
        idx_by_depth = np.argsort(points_camera[:,2])
        gs = gs
        gs.rearange_gs(idx_by_depth)
        projected_pixels=pixels[idx_by_depth]
        self.block_xy= block_xy

        gs.calc_cov3d()
        gs.calc_cov2d(cam)
        # gs.get_color(3,xyz = sorted_gs.xyz,camera_position = cam.t)
        self.bounding_box = gs.get_rect(cam)

        tile_coords_range = [(x_idx,y_idx) for x_idx in range(tiles[0], tiles[1]) for y_idx in range(tiles[0], tiles[1])]
        self.tiles = {(x_idx,y_idx): self.get_current_tile_params(gs,x_idx,y_idx,projected_pixels) for x_idx,y_idx in tile_coords_range}
        self.rendered_image = np.ones((image_size[0],image_size[1],3))
        

    def calc_pixel_value(self,tile_params,pixel):
        d = tile_params['projection']   - pixel 
        power = -0.5 * (tile_params['conic'][:,0] * d[:,0] * d[:,0] + tile_params['conic'][:,2] * d[:,1] * d[:,1]) - tile_params['conic'][:,1] * d[:,0] * d[:,1]
        alpha = np.minimum(0.99,  tile_params['conic'][:,3]*np.exp(power))
        idx_to_keep = (alpha>1/255) & (power <= 0)
        return self.sum_all_gs_in_tile(alpha[idx_to_keep],tile_params['color'][idx_to_keep])

    def get_pixels_in_tile(self,pix_start_end):
        xv,yv = np.meshgrid(range(pix_start_end[0][0],pix_start_end[1][0]),range(pix_start_end[0][1],pix_start_end[1][1]))
        return np.column_stack((np.reshape(xv,xv.shape[0]*xv.shape[1]),np.reshape(yv,xv.shape[0]*xv.shape[1])))
    
    def render_image(self):
        [self.calc_pixels_value_in_tile(tile) for tile in self.tiles]
        return self.rendered_image


    def calc_pixels_value_in_tile(self,tile):
        pix_start_end = (np.array(tile) - 1)*self.block_xy[0],(np.array(tile))*self.block_xy[0]
        pixels_in_tile = self.get_pixels_in_tile(pix_start_end)
        if len(self.tiles[tile]['projection']) > 0:
            for pixel in pixels_in_tile:
                pixel_value,T = self.calc_pixel_value(self.tiles[tile],pixel)
                self.rendered_image[pixel[1],pixel[0]] = pixel_value + T*np.array([1,1,1])




    def get_current_tile_params(self,gs,x_idx,y_idx,projected_pixels):
        count_within_bounds = np.where(np.sum((self.bounding_box[0] <= [x_idx,y_idx]  ) & (self.bounding_box[1] >= [x_idx,y_idx] ), axis = 1) == 2)[0]
        tile_params = {'xyz': gs.xyz[count_within_bounds], 'conic':gs.conic[count_within_bounds],
                    'color': gs.color[count_within_bounds], 'opacity' : gs.opacity[count_within_bounds],
                    'projection': projected_pixels[count_within_bounds,0:2]}
        return tile_params

    def sum_all_gs_in_tile(self,alpha,color): 
        T = 1
        clr = [0,0,0]
        for trans,col in zip(alpha,color): 
            clr += col*trans*T
            T = T*(1-trans)
            if T < 0.0000001:
                break
        return clr,T
    

        