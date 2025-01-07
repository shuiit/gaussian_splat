
from Joint import Joint
import numpy as np
from Bone import Bone
import Plotter
class Skeleton():
    def __init__(self):
        self.bones = []
        self.joints = {}
         

    def initilize_joints(self,joints):
        [self.add_joint(joint_name,transltation) for joint_name, transltation in joints.items()]
        


    def add_joint(self,joint_name,transltation):
        self.joints[joint_name] = Joint(joint_name,translation_from_parent = transltation ,joint_without_bone = (len(joint_name.split('no_bone')) >1))

    def connect_parent_child(self,parent_child):
        for joint_name,children_names in parent_child.items():
            [self.joints[joint_name].add_child(self.joints[child_name]) for child_name in children_names]


    def add_bone_to_joint(self):
        self.bones = []
        for joint_name in self.joints.keys():
            joint = self.joints[joint_name]
            if ((joint.parent != None) and (joint.joint_without_bone == False)) and (joint.parent.joint_without_bone == False):
                self.add_bone(self.joints[joint_name].parent, self.joints[joint_name])
                self.bones.append(self.joints[joint_name].parent.name)

    def update_bone(self):
        [self.joints[joint_name].bone.update_bone() for joint_name in self.joints.keys() if self.joints[joint_name].bone != None]

    def add_bone(self, parent_joint, child_joint):
        bone = Bone(parent_joint, child_joint)
        parent_joint.bone = bone



    def build_skeletone(self,joints,parent_child):
        for joint_name,children_names in parent_child.items():
            [self.joints[joint_name].add_child(self.joints[child_name]) for child_name in children_names]

      
    def update_local_roattion(self,local_rotation):
        [self.joints[joint_name].set_local_rotation(rotation) for joint_name,rotation in local_rotation.items()]

    def update_global_rotation(self, **kwargs):
        [self.joints[joint_name].get_global_transformation(**kwargs) for joint_name in self.joints.keys()]


    def get_global_point_skeleton_branch(self,point = [0,0,0,1]):
        [self.joints[joint_name].get_global_point(point) for joint_name in self.joints.keys()]
    

    def calculate_weight(self,points, th = 1, constant_weight = False, **kwargs):
        if constant_weight in self.bones:
            weight_map = np.zeros((points.shape[0],len(self.bones)))
            weight_map[:,self.bones.index(constant_weight)] = 1
        else:
            weight = np.vstack([ self.joints[joint_name].bone.calculate_dist_from_bone(points[:,0:3], **kwargs) for joint_name in self.joints.keys() if self.joints[joint_name].bone is not None]).T
            weight_map = (1/weight)/np.sum((1/weight),1)[:,np.newaxis]
            weight_map[weight_map > th] = 1
        return weight_map
    

    def plot_skeleton(self,fig,marker_dict,line_dict):
        
        for joint in self.joints.values():
            marker_dict['color'] = joint.color
            line_dict['color'] = joint.color
            if joint.bone is not None:
                Plotter.scatter3d(fig,joint.bone.bone_points,joint.bone.bone_points_names[0],mode = 'lines',line_dict= line_dict)
                Plotter.scatter3d(fig,joint.bone.bone_points,joint.bone.bone_points_names[0],mode = 'markers',marker_dict= marker_dict)
