import sys
import os
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import Json_Parser
from network import CoverNet
from dataset_covernet import NuSceneDataset_CoverNet
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory


class CoverNet_test:
    def __init__(self, config_file, trained_model_path, dataset_train_mode, dataset_batch_size, num_modes, verbose):
        self.parser = Json_Parser(config_file)
        self.config = self.parser.load_parser()
        self.model_weight_path = trained_model_path
        self.traj_set_path = self.config['LEARNING']['trajectory_set_path']
        self.device = torch.device(self.config['LEARNING']['device'] if torch.cuda.is_available() else 'cpu')
        self.num_modes = num_modes

        self.backbone = ResNetBackbone('resnet50')
        self.model = CoverNet(self.backbone, num_modes=num_modes)
        self.model.load_state_dict(torch.load(self.model_weight_path))

        self.trajectory_set = pickle.load(open(self.traj_set_path, 'rb'))
        
        self.nuscenes = NuScenes(self.config['DATASET']['dataset_str'], dataroot=self.config['DATASET']['dataset_path'], verbose=verbose)
        self.helper = PredictHelper(self.nuscenes)
        self.dataset = NuSceneDataset_CoverNet(train_mode=dataset_train_mode, config_file_name=config_file, verbose=verbose)
        self.rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=1)    
        
        self.num_future_hist = self.config['HISTORY']['num_future_hist']         

        self.resolution = self.config['PREPROCESS']['resolution']         
        self.meters_ahead = self.config['PREPROCESS']['meters_ahead']
        self.meters_behind = self.config['PREPROCESS']['meters_behind']
        self.meters_left = self.config['PREPROCESS']['meters_left']
        self.meters_right = self.config['PREPROCESS']['meters_right'] 


    def draw_traj_set(self):
        for i in range(len(self.trajectory_set)):
            t = np.array(self.trajectory_set[i][:])
            plt.plot(t[:,0],t[:,1])
        plt.show()


    def overlay_result(self, img, path_set, color = (255, 255, 0)): 
        # print("path_set : ",path_set)

        if type(path_set) == list:
            print("Path_set type is List!")
            print(path_set)
            path_set = np.array(path_set)
            # path to image coordinate transform (path is vehicle center coordinated)
            path_u = (-path_set[:,1] + self.meters_left)/self.resolution
            path_v = (self.meters_ahead -  path_set[:,0])/self.resolution
            for i in range(len(path_v)):
                cv2.circle(img, (int(path_u[i]), int(path_v[i])), 5, color, -1)
                if i > 0:
                    cv2.line(img,(prev_path_u, prev_path_v),(path_u[i], path_v[i]),color, 2 )
                prev_path_u = int(path_u[i])
                prev_path_v = int(path_v[i])
                    
        elif type(path_set) == torch.Tensor:
            print("Path_set type is Tensor!")
            for path_modes in path_set:
                for path in path_modes:
                    # path to image coordinate transform (path is vehicle center coordinated)
                    path_u = (-path[1] + self.meters_left)/self.resolution
                    path_u = path_u.int().detach().cpu().numpy()
                    path_v = (self.meters_ahead -  path[0])/self.resolution
                    path_v = path_v.int().detach().cpu().numpy()
                    for i in range(len(path_v)):
                        cv2.circle(img, (path_u[i], path_v[i]), 5, color, -1)
                        if i > 0:
                            cv2.line(img,(prev_path_u, prev_path_v),(path_u[i], path_v[i]),color, 2 )
                        prev_path_u = path_u[i]
                        prev_path_v = path_v[i]
        return img

    # def convert_to_pixel_coords(self,
    #                             location: Tuple[float, float],
    #                             center_of_image_in_global: Tuple[float, float],
    #                             center_of_image_in_pixels: Tuple[float, float],
    #                             resolution: float = 0.1) -> Tuple[int, int]:
    #     x, y = location
    #     x_offset = (x - center_of_image_in_global[0])
    #     y_offset = (y - center_of_image_in_global[1])

    #     x_pixel = x_offset / resolution

    #     # Negate the y coordinate because (0, 0) is ABOVE and to the LEFT
    #     y_pixel = -y_offset / resolution

    #     row_pixel = center_of_image_in_pixels[0] + y_pixel
    #     column_pixel = center_of_image_in_pixels[1] + x_pixel

    #     # row_pixel = list(map(int, row_pixel))   
    #     # column_pixel = list(map(int, column_pixel))   
    #     print("location : ", location)
    #     print("center_of_image_in_global : ", center_of_image_in_global)

    #     print("x_offset : ", x_offset)
    #     print("row_pixel : ", row_pixel)
    #     print("column_pixel : ", column_pixel)

    #     return row_pixel, column_pixel


    # def draw_lanes_on_image(self, agent_pos: List,
    #                               image: np.ndarray,
    #                               lanes: Dict[str, List[Tuple[float, float, float]]],
    #                               mode):
    #     if mode =="pred":
    #         color = (255, 255, 0) # yellow
    #     elif mode =="label":
    #         color = (204, 0, 204) # violet
    #     elif mode =="true":
    #         color = (0, 0, 255) # red

    #     image_side_length = 2 * max(self.meters_ahead, self.meters_behind,
    #                                 self.meters_left, self.meters_right)
    #     image_side_length_pixels = int(image_side_length / self.resolution)
    #     print("image_side_length_pixels : " , image_side_length_pixels)

    #     agent_pixels = [int(image_side_length_pixels / 2), int(image_side_length_pixels / 2)]
    #     agent_global_coords = (agent_pos[0], agent_pos[1])

    #     for poses_along_lane in lanes.values():
    #         for start_pose, end_pose in zip(poses_along_lane[:-1], poses_along_lane[1:]):
    #             start_pixels = self.convert_to_pixel_coords(start_pose[:2], agent_global_coords,
    #                                                agent_pixels, self.resolution)
    #             end_pixels = self.convert_to_pixel_coords(end_pose[:2], agent_global_coords,
    #                                              agent_pixels, self.resolution)

    #             start_pixel = (start_pixels[1], start_pixels[0])
    #             end_pixel = (end_pixels[1], end_pixels[0])
    #             print(start_pixel)
    #             cv2.line(image, start_pixel, end_pixel, color, thickness=3)
            
    #     return image

    def traject_to_lane(self, traject):
        str = 'lane'
        x = traject[:,0]
        y = traject[:,1]
        l = []
        for i in range(len(x)):
            t = (x[i],y[i],0)
            l.append(t)
        dic = {str:l}
        
        return dic


    def trajectories_to_lane(self, traject):
        ## Make trajectory to Dict[str, List[Tuple[float, float, float]]] form 
        output = {}

        for i in range(np.shape(traject)[0]):
            str = '{}'.format(i)
            x = traject[i][:,0]
            y = traject[i][:,1]
            
            lane = []
            for j in range(len(x)):
                temp = (x[j], y[j], 0.0)
                lane.append(temp)

            output[str] = lane

        # print("output : ", output)
        return output


    def run(self):
        data = self.dataset.__getitem__(10)

        self.model.eval()
        ego_pos = data['ego_cur_pos']
        print("ego_pos : ", ego_pos)
        img_raw =  data['img'].detach().cpu().numpy()
        img_raw = np.ascontiguousarray(img_raw, dtype=np.uint8)

        img_tensor = torch.Tensor(data['img']).unsqueeze(0).to(self.device)
        agent_state_tensor = torch.Tensor(data['ego_state'].tolist()).to(self.device)
        agent_state_tensor = torch.squeeze(agent_state_tensor, 1)

        prediction = self.model(img_tensor, agent_state_tensor)
        pred = F.softmax(prediction,dim=-1)
        # print("pred.argmax() :", pred.argmax())
        label = data['label']
        # print("label :", label)


        # pred_traj = self.trajectory_set[pred.argsort(descending=True)[:1]]       
        pred_traj = self.trajectory_set[pred.argmax()]
        label_traj = self.trajectory_set[label]
        self.overlay_result(img_raw, pred_traj) 

        # pred_lane = self.traject_to_lane(np.array(pred_traj))
        # label_lane = self.traject_to_lane(np.array(label_traj))
        # real_lane = self.traject_to_lane(real)
        # print("pred_lane :", pred_lane)
        # print("label_lane :", label_lane)


        # result_img = self.draw_lanes_on_image(ego_pos, img_raw, pred_lane, 'pred')
        # result_img = self.draw_lanes_on_image(ego_pos, img_raw, label_lane, 'label')
        # result_img = self.draw_lanes_on_image(img_raw, real_lane, 'true', 0.1)
        # cv2.imshow("Result Comparison", result_img)


        ## trajectory set plot on axis 
        # self.draw_traj_set()

        traj_sets = []
        for i in range(len(self.trajectory_set)):
            t = np.array(self.trajectory_set[i][:])
            traj_sets.append(t)

        traj_sets = self.trajectories_to_lane(np.array(traj_sets))
        # print(traj_sets)
        # traj_sets_img = self.draw_lanes_on_image(ego_pos, img_raw, traj_sets, 'pred')
        # cv2.imshow("Trajectory_sets", traj_sets_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows


if __name__ == "__main__":
    covernet_test = CoverNet_test(config_file='./covernet_config.json', 
                                 trained_model_path='./result/model/20211025-20_35_49/epoch_4.pth',
                                 dataset_train_mode=False, 
                                 dataset_batch_size=4,
                                 num_modes=64, 
                                 verbose=False)
    covernet_test.run()
