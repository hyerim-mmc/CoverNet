import os
import json
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.input_representation.interface import Combinator
from pyquaternion.quaternion import Quaternion
from typing import List

class Json_Parser:
    def __init__(self, file_name):
        with open(file_name) as json_file:
            self.config = json.load(json_file)

    def load_parser(self):
        return self.config


def data_filter(data):
	for i in range(len(data)):
		nan_check = np.isnan(data[i])
		if nan_check:
			data[i] = 0
			
	return data

def save_imgs(self, dataset, type, input_repr):
	print("starting to save maps")

	for i, _ in enumerate(dataset): 
		instance_token_img, sample_token_img = dataset[i].split('_')
		
		folder_path = os.path.join(self.dataroot, 'saved_img', type)
		if not os.path.exists(folder_path):
			os.makedirs(folder_path, exist_ok=True)

		file_path = os.path.join(folder_path,"img_{0}.jpg".format(i))

		instance_token_img, sample_token_img = dataset[i].split('_')
		img = input_repr.make_input_representation(instance_token_img, sample_token_img)
		im = Image.fromarray(img)
		im.save(file_path)
	
		print("Img saving process : [{0}/{1}] completed".format(i, len(dataset)),end='\r')
	print("done saving imgs")


def save_maps(self, type, map, idx):
	folder_path = os.path.join(self.dataroot, 'saved_map', type)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path, exist_ok=True)
	file_path = os.path.join(folder_path,"maps_{0}.jpg".format(idx))

	plt.savefig(file_path)	
	print("done saving map_{}".format(idx))


def get_pose_from_annot(annotation) -> list:
    x, y, _ = annotation['translation']
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(annotation['rotation'])))
    
    return [x, y, yaw]

def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
	
def get_pose(annotation_list) -> np.ndarray:
    return np.array([get_pose_from_annot(ann) for ann in annotation_list])


def rotation_global_to_local(yaw) -> np.ndarray:
    return np.array([[ np.cos(yaw), -np.sin(yaw)], \
                [np.sin(yaw), np.cos(yaw)]])


def angle_mod_2pi(angle):
	return (angle + np.pi) % (2.0 * np.pi) - np.pi


def convert_global_to_local_forhistory(global_pose_origin, global_poses) -> np.ndarray:
	R_global_to_local = rotation_global_to_local(global_pose_origin[2])
	t_global_to_local = - R_global_to_local @ global_pose_origin[:2]

	output = []
	for pose in global_poses:
		local_xy = R_global_to_local @ pose[:2] + t_global_to_local
		local_yaw = angle_mod_2pi(pose[2] - global_pose_origin[2])
		temp = [local_xy[0], local_xy[1], local_yaw]
		output.append(temp)

	return np.array(output)


def convert_global_to_local_forpose(global_pose_origin, global_pose) -> np.ndarray:
    R_global_to_local = rotation_global_to_local(global_pose_origin[2])
    t_global_to_local = - R_global_to_local @ global_pose_origin[:2]

    local_xy  = R_global_to_local @ global_pose[:2] + t_global_to_local 
    local_yaw = angle_mod_2pi(global_pose[2] - global_pose_origin[2])
    output = [local_xy[0], local_xy[1], local_yaw]

    return np.array(output)