
from Joint import Joint
import numpy as np

class Skeleton():
    def __init__(self,joints,parent_child):
        self.build_skeletone(joints,parent_child)
        self.hirarchy = self.get_skeletone_hirarchy(self.joints['root'])[1]

    def build_skeletone(self,joints,parent_child):
        self.joints = {joint_name : Joint(joint_name,translation_from_parent = transltation) for joint_name, transltation in joints.items()} 
        for joint_name,children_names in parent_child.items():
            [self.joints[joint_name].add_child(self.joints[child_name]) for child_name in children_names]

    def get_skeletone_hirarchy(self,joint,all_points = []):
        if len(joint.child) == 0:
            return [],all_points
        for child in joint.child:
            branch,all_points = self.get_skeletone_hirarchy(child)
            branch.append(child.name)
            if joint.parent == None:
                branch.append('root')
                all_points.append(branch[::-1])
        return branch,all_points
    
    def update_local_roattion(self,local_rotation):
        [self.joints[joint_name].set_local_rotation(rotation) for joint_name,rotation in local_rotation.items()]

    def update_global_rotation(self):
        [self.joints[joint_name].get_global_transformation() for joint_name in self.joints.keys()]


    def get_global_point_skeleton_branch(self,point = [0,0,0,1]):
        all_points = []
        for branch in self.hirarchy:
            all_points.append(np.vstack([self.joints[joint_name].get_global_point(point) for joint_name in branch]))
        return all_points