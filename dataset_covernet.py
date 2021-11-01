import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import utils
import torch
import intersection_dataload
import numpy as np

from utils import Json_Parser
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer


class NuSceneDataset_CoverNet(Dataset):
    def __init__(self, train_mode, config_file_name, layers_list=None, color_list=None, verbose=True):
        super().__init__()
        parser = Json_Parser(config_file_name)
        config = parser.load_parser()
        
        self.verbose = verbose
        self.device = torch.device(config['LEARNING']['device'] if torch.cuda.is_available() else 'cpu')
        self.dataroot = config['DATASET']['dataset_path']
        self.intersection_use= config['DATASET']['intersection_use']        # only available for mini_dataset
        self.nuscenes = NuScenes(config['DATASET']['dataset_str'], dataroot=self.dataroot, verbose=self.verbose)
        self.helper = PredictHelper(self.nuscenes)

        self.set = config['DATASET']['set']
        self.train_mode = train_mode
        if self.set == 'train':
            self.mode = 'train'
            self.train_set = get_prediction_challenge_split("train", dataroot=self.dataroot)
            self.val_set = get_prediction_challenge_split("train_val", dataroot=self.dataroot)
        else:            
            self.mode = 'mini'
            self.train_set = get_prediction_challenge_split("mini_train", dataroot=self.dataroot)
            self.val_set = get_prediction_challenge_split("mini_val", dataroot=self.dataroot)

            if self.intersection_use:
                self.train_set = intersection_dataload.token_save(self.train_set)
                self.val_set = intersection_dataload.token_save(self.val_set)
                
        if layers_list is None:
            self.layers_list = config['PREPROCESS']['img_layers_list']
        if color_list is None:
            self.color_list = []
            for i in range(len(self.layers_list)):
                self.color_list.append((255,255,255))

        self.resolution = config['PREPROCESS']['resolution']         
        self.meters_ahead = config['PREPROCESS']['meters_ahead']
        self.meters_behind = config['PREPROCESS']['meters_behind']
        self.meters_left = config['PREPROCESS']['meters_left']
        self.meters_right = config['PREPROCESS']['meters_right'] 

        self.num_past_hist = int(config['HISTORY']['num_past_hist']/2)
        self.num_future_hist = int(config['HISTORY']['num_future_hist']/2)

        self.static_layer = StaticLayerRasterizer(helper=self.helper, 
                                            layer_names=self.layers_list, 
                                            colors=self.color_list,
                                            resolution=self.resolution, 
                                            meters_ahead=self.meters_ahead, 
                                            meters_behind=self.meters_behind,
                                            meters_left=self.meters_left, 
                                            meters_right=self.meters_right)
        self.agent_layer = AgentBoxesWithFadedHistory(helper=self.helper, 
                                                seconds_of_history=self.num_past_hist)
        self.input_repr = InputRepresentation(static_layer=self.static_layer, 
                                        agent=self.agent_layer, 
                                        combinator=Rasterizer())     

        self.show_imgs = config['PREPROCESS']['show_imgs']
        self.save_imgs = config['PREPROCESS']['save_imgs']

        self.num_max_agent = config['PREPROCESS']['num_max_agent']
        
        self.traj_set_path = config['LEARNING']['trajectory_set_path']
        self.trajectories_set =torch.Tensor(pickle.load(open(self.traj_set_path, 'rb')))

        if self.save_imgs:
            if self.train_mode:
                utils.save_imgs(self, self.train_set, self.set + 'train', self.input_repr)
            else:
                utils.save_imgs(self, self.val_set, self.set + 'val', self.input_repr)
        
  
    def __len__(self):
        if self.train_mode:
            return len(self.train_set)
        else:
            return len(self.val_set)


    def get_label(self, trajectory_set, ground_truth):
        return self.mean_pointwise_l2_distance(trajectory_set, ground_truth)


    def mean_pointwise_l2_distance(self, lattice: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Computes the index of the closest trajectory in the lattice as measured by l1 distance.
        :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
        :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
        :return: Index of closest mode in the lattice.
        """
        stacked_ground_truth = ground_truth.repeat(lattice.shape[0], 1, 1)
        return torch.pow(lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()

    def __getitem__(self, idx):
        if self.train_mode:
            self.dataset = self.train_set
        else:
            self.dataset = self.val_set

        #################################### State processing ####################################
        ego_instance_token, ego_sample_token = self.dataset[idx].split('_')
        ego_annotation = self.helper.get_sample_annotation(ego_instance_token, ego_sample_token)

        ego_pose = np.array(utils.get_pose_from_annot(ego_annotation))
        ego_vel = self.helper.get_velocity_for_agent(ego_instance_token, ego_sample_token)
        ego_accel = self.helper.get_acceleration_for_agent(ego_instance_token, ego_sample_token)
        ego_yawrate = self.helper.get_heading_change_rate_for_agent(ego_instance_token, ego_sample_token)
        [ego_vel, ego_accel, ego_yawrate] = utils.data_filter([ego_vel, ego_accel, ego_yawrate])                # Filter unresonable data (make nan to zero)
        ego_states = np.array([[ego_vel, ego_accel, ego_yawrate]])

        # GLOBAL history
        future_poses_m = np.zeros((self.num_future_hist, 3))
        future = self.helper.get_future_for_agent(instance_token=ego_instance_token, sample_token=ego_sample_token, 
                                            seconds=int(self.num_future_hist/2), in_agent_frame=False, just_xy=False)
        num_future_mask = len(future)
        future_poses_m[:len(future)] = utils.get_pose(future)

        # Get label
        gt_tensor = torch.Tensor(future_poses_m[:,:2]).unsqueeze(0)
        trajectories_tensor = self.trajectories_set[:,:self.num_future_hist]
        label = self.get_label(trajectories_tensor, gt_tensor)


        #################################### Image processing ####################################
        img = self.input_repr.make_input_representation(instance_token=ego_instance_token, sample_token=ego_sample_token)
        if self.show_imgs:
            plt.figure('input_representation')
            plt.imshow(img)
            plt.show()

        img = torch.Tensor(img).permute(2,0,1).to(device=self.device)


        return {'img'                  : img,                          # Type : torch.Tensor
                'ego_cur_pos'          : ego_pose,                                                    # Type : np.array([global_x,globa_y,global_yaw])                        | Shape : (3, )
                'ego_state'            : ego_states,                   # Type : np.array([[vel,accel,yaw_rate]]) --> local(ego's coord)   |   Unit : [m/s, m/s^2, rad/sec]    
                'future_global_ego_pos': future_poses_m,               # Type : np.array([global_x, global_y, global_yaw]) .. ground truth data
                'num_future_mask'      : num_future_mask,              # a number for masking future history
                'label'                : label                         # calculated label data from preprocessed_trajectory_set using ground truth data
                }

    
# if __name__ == '__main__':
#     train_dataset = NuSceneDataset_CoverNet(train_mode=True, config_file_name='./covernet_config.json', verbose=True)
#     print(len(train_dataset))
#     print(train_dataset.__len__())

#     val_dataset = NuSceneDataset_CoverNet(train_mode=False, config_file_name='./covernet_config.json', verbose=True)
#     print(len(val_dataset))
#     print(val_dataset.__len__())
