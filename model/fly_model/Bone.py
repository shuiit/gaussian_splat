import numpy as np


class Bone:
    def __init__(self,parent_joint,child_joint):
  
        self.parent = parent_joint
        self.child = child_joint
        self.bone_points_names = [parent_joint.name,child_joint.name]
        self.length = np.linalg.norm((np.array(parent_joint.global_origin) - np.array(child_joint.global_origin)))
        self.direction = (np.array(parent_joint.global_origin) - np.array(child_joint.global_origin))/self.length
        self.bone_points = np.vstack([parent_joint.global_origin,child_joint.global_origin])





    def calculate_dist_from_bone(self,points):
        
        points_to_bone_origin = points - self.bone_points[0] # vector between the 3d points and the bone origin (not nomalized)
        bone_vector = self.bone_points[1] - self.bone_points[0] # The vector representing the bone (from origin to end, not normalized)
        t = np.dot(points_to_bone_origin, bone_vector) / self.length**2 # Project 'points_to_bone_origin' onto 'bone_vector' to determine how far along the bone each point is.
                                                                        # Normalize by the bone's squared length to express this projection in units of the bone's length.
        t = np.clip(t, 0, 1)[:,np.newaxis] # clip to get the closest point, if its on the bone, its between [0,1] else its outside the bone
        closest_point = self.bone_points[0] + t * bone_vector # determin the closest point, if its outside the bone take the endpoint as the closest point
        return np.linalg.norm(points - closest_point,axis  =1) # calculte the distance between the 3d point and the closest point on the bone
        
