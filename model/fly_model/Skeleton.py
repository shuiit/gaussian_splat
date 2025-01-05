
from Joint import Joint
import numpy as np
from Bone import Bone
class Skeleton():
    def __init__(self):
        self.bones = []
        self.joints = {} 


        # self.build_skeletone(joints,parent_child)
        # self.root_name = root_name
        # self.hirarchy = self.get_skeletone_hirarchy(self.joints[root_name])[1]

    def initilize_joints(self,joints):
        [self.add_joint(joint_name,transltation) for joint_name, transltation in joints.items()]
        


    def add_joint(self,joint_name,transltation):
        self.joints[joint_name] = Joint(joint_name,translation_from_parent = transltation ,joint_without_bone = (len(joint_name.split('no_bone')) >1))

    def connect_parent_child(self,parent_child):
        for joint_name,children_names in parent_child.items():
            [self.joints[joint_name].add_child(self.joints[child_name]) for child_name in children_names]


    def add_bone_to_joint(self):
        for joint_name in self.joints.keys():
            joint = self.joints[joint_name]
            if ((joint.parent != None) and (joint.joint_without_bone == False)) and (joint.parent.joint_without_bone == False):
                self.add_bone(self.joints[joint_name].parent, self.joints[joint_name])


        # no_bone = (len(joint.joint_name.split('no_bone')) > 1)
        # self.joints.append(Joint(joint.joint_name,translation_from_parent = joint.transltation))
        # self.joints[-1].get_global_point()
        # if (joint.parent is not None) & (no_bone == False):
        #     self.add_bone(joint.parent, joint)


    def add_bone(self, parent_joint, child_joint):
        bone = Bone(parent_joint, child_joint)
        parent_joint.bone = bone





    def build_skeletone(self,joints,parent_child):
        for joint_name,children_names in parent_child.items():
            [self.joints[joint_name].add_child(self.joints[child_name]) for child_name in children_names]

    # def get_skeletone_hirarchy(self,joint, all_branches=None):
    #     if all_branches is None:
    #         all_branches = []
    #     if len(joint.child) == 0:
    #         return [],all_branches
    #     for child in joint.child:
    #         branch,all_branches = self.get_skeletone_hirarchy(child,all_branches)
    #         branch.append(child.name)
    #         if joint.parent == None:
    #             branch.append(joint.name)
    #             all_branches.append(branch[::-1])
    #     return branch,all_branches
    
    def update_local_roattion(self,local_rotation):
        [self.joints[joint_name].set_local_rotation(rotation) for joint_name,rotation in local_rotation.items()]

    def update_global_rotation(self, **kwargs):
        [self.joints[joint_name].get_global_transformation(**kwargs) for joint_name in self.joints.keys()]


    def get_global_point_skeleton_branch(self,point = [0,0,0,1]):
        [self.joints[joint_name].get_global_point(point) for joint_name in self.joints.keys()]
    
