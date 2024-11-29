import numpy as np
class Render():
    def __init__(self,gs,cam,tiles = [1,10],block_xy = [16,16], image_size = [160,160],gaus3d = True,filtersze =  0.7071067811865476):
        """
        Initializes the Render class, setting up the Gaussian splats, camera projections, 
        depth sorting, and tiling for efficient rendering.

        Args:
            gs (GaussianSplat): Gaussian splatting object with attributes like xyz, color, conic.
            cam (Camera): Camera object to project 3D points to 2D image plane.
            tiles (list): List defining tile boundaries on x and y axis for rendering.
            block_xy (list): Tile block size for x and y in pixels.
            image_size (list): Dimensions of the output image [height, width].
        """
        self.gaus3d = gaus3d
        self.filtersze = filtersze
        pixels = cam.project_with_proj_mat(gs.xyz)
        self.points_camera = cam.rotate_world_to_cam(cam.homogenize_coordinate(gs.xyz))
        idx_by_depth = np.argsort(self.points_camera[:,2])
        gs.rearange_gs(idx_by_depth)
        projected_pixels=pixels[idx_by_depth]
        self.block_xy= block_xy
        gs.calc_cov3d()
        gs.calc_cov2d(cam)
        gs.calculate_T_2d(cam)
        
        if gaus3d == False:
            gs.radius = gs.radius_2d
        self.bounding_box = gs.get_rect(cam)

        tile_coords_range = [(x_idx,y_idx) for x_idx in range(tiles[0], tiles[1]) for y_idx in range(tiles[0], tiles[1])]
        self.tiles = {(x_idx,y_idx): self.get_current_tile_params(gs,x_idx,y_idx,projected_pixels) for x_idx,y_idx in tile_coords_range}
        self.rendered_image = np.ones((image_size[0],image_size[1],3))
        self.depth = np.ones((image_size[0],image_size[1],3))
        

    def intersection_point(self,pixel,T):
        # T is a transformation matrix from tangent plane to NDC to pixels.
        # We start from screen space (pixel coordinates), and transform them to a different coordinate space
        # using the transformation matrix T. Then we compute the intersection of the transformed directions 
        # corresponding to the xz and yz planes. The result is normalized by the depth (z-coordinate) to 
        # project the intersection point onto the tangent plane.   
        k = -T[..., 0] + pixel[0] * T[...,2]
        l = -T[..., 1] + pixel[1] * T[..., 2]
        points = np.cross(k, l, axis=-1)
        return points[..., :2] / points[..., -1:]
    
    def calc_pixel_value(self,tile_params,pixel):
        """
        Calculates the pixel value based on Gaussian splats within the tile using the 
        Gaussian projection and alpha blending.

        Args:
            tile_params (dict): Dictionary containing Gaussian splat parameters for the tile.
            pixel (np.array): 2D pixel coordinate in the image.

        Returns:
            tuple: Color value for the pixel and remaining transparency.
        """
        
        d = tile_params['projection'][:,0:2]   - pixel 
        # power is the gaussian distirbuition. we get the amplitude of each gaussian that impact this pixel. 
        if self.gaus3d == True: 
            power = -0.5 * (tile_params['conic'][:,0] * d[:,0] * d[:,0] + tile_params['conic'][:,2] * d[:,1] * d[:,1]) - tile_params['conic'][:,1] * d[:,0] * d[:,1]
            
        else:
            s = self.intersection_point(pixel,tile_params['T']) # get the ray from the pixel to the gaussian as seen in the tangent plane
            dist3d = np.sum(s * s, axis=-1) # a 2d gaussian is defined G(u,v) = exp(-(u^2+v^2)/2*sigma^2), here we calculate u^2 + v^2 (everything in object plane)
            dist2d =  (1 / self.filtersze) ** 2 * np.linalg.norm(pixel - tile_params['center'][ :, :2], axis=-1) ** 2
            # Compute the squared screen-space distance (x - mu)^2, where mu is the Gaussian's projected mean 
            # and x is the pixel position.
            # When a 2D Gaussian is observed from a slanted viewpoint, it can degenerate into a line in screen space. 
            # This can result in rasterization artifacts where the Gaussian is under-sampled or missed entirely.
            # To address this, we define a fallback Gaussian centered at the projected mean of the original Gaussian 
            # in screen space. We use the smaller of dist3d (tangent-plane distance) and dist2d (screen-space distance) 
            # to ensure coverage.
            # If dist2d < dist3d, it indicates that distortions in the tangent plane (e.g., due to perspective effects) 
            # are significant. In this case, the screen-space Gaussian serves as a low-pass filter to stabilize the rendering.
            power = -0.5 * np.minimum(dist3d, dist2d)  

        alpha = np.minimum(0.99,  tile_params['conic'][:,3]*np.exp(power))
        idx_to_keep = (alpha>=1/255) & (power <= 0)
        image,T = self.sum_all_gs_in_tile(alpha[idx_to_keep],tile_params['color'][idx_to_keep])
        depth,T = self.sum_all_gs_in_tile(alpha[idx_to_keep],tile_params['cam_coord'][idx_to_keep,2])

        return image,T,np.array(depth)

    def get_pixels_in_tile(self,pix_start_end):
        """
        Generates a grid of pixels within a specified tile range.

        Args:
            pix_start_end (tuple): Starting and ending pixel coordinates of the tile.

        Returns:
            np.array: Array of pixel coordinates within the tile.
        """
        xv,yv = np.meshgrid(range(pix_start_end[0][0],pix_start_end[1][0]),range(pix_start_end[0][1],pix_start_end[1][1]))
        return np.column_stack((np.reshape(xv,xv.shape[0]*xv.shape[1]),np.reshape(yv,xv.shape[0]*xv.shape[1])))
    
    def render_image(self):
        """
        Renders the final image by iterating over each tile and calculating pixel values.

        Returns:
            np.array: The rendered image as a 3D numpy array (height, width, color channels).
        """
        [self.calc_pixels_value_in_tile(tile) for tile in self.tiles]
        return self.rendered_image


    def calc_pixels_value_in_tile(self,tile):
        """
        Calculates and assigns values for each pixel in a specified tile.

        Args:
            tile (tuple): Tile coordinate in the tile grid.
        """
        pix_start_end = (np.array(tile) - 1)*self.block_xy[0],(np.array(tile))*self.block_xy[0]
        pixels_in_tile = self.get_pixels_in_tile(pix_start_end)
        if len(self.tiles[tile]['projection']) > 0:
            for pixel in pixels_in_tile:
                pixel_value,temp_alpha,depth = self.calc_pixel_value(self.tiles[tile],pixel)
                self.rendered_image[pixel[1],pixel[0]] = pixel_value + temp_alpha*np.array([1,1,1])
                self.depth[pixel[1],pixel[0]] = depth 
        




    def get_current_tile_params(self,gs,x_idx,y_idx,projected_pixels,T = None,center = None):
        """
        Gets Gaussian splat parameters for the specified tile.

        Args:
            gs (GaussianSplat): Gaussian splatting object with splat parameters.
            x_idx (int): Tile index in the x direction.
            y_idx (int): Tile index in the y direction.
            projected_pixels (np.array): Projected 2D coordinates of points.

        Returns:
            dict: Dictionary of parameters like xyz, conic, color, opacity, and projection for the tile.
        """
        count_within_bounds = np.where(np.sum((self.bounding_box[0] <= [x_idx,y_idx]  ) & (self.bounding_box[1] >= [x_idx,y_idx] ), axis = 1) == 2)[0]
        tile_params = {'xyz': gs.xyz[count_within_bounds], 'conic':gs.conic[count_within_bounds],
                    'color': gs.color[count_within_bounds], 'opacity' : gs.opacity[count_within_bounds],
                    'projection': projected_pixels[count_within_bounds,0:3],'cam_coord': self.points_camera[count_within_bounds]}
        if self.gaus3d != True:
             tile_params['T'] = gs.T[count_within_bounds]
             tile_params['center'] = gs.center[count_within_bounds]
        return tile_params

    def sum_all_gs_in_tile(self,alpha,color): 
        """
        Blends the colors of all Gaussian splats within a tile based on their alpha values.

        Args:
            alpha (np.array): Alpha values of the Gaussian splats.
            color (np.array): Color values of the Gaussian splats.

        Returns:
            tuple: Blended color value and remaining transparency.
        """
        T = 1
        clr = [0,0,0]
        for trans,col in zip(alpha,color): 
            clr += col*trans*T
            T = T*(1-trans)
            if T < 0.0001:
                break
        return clr,T
    

        