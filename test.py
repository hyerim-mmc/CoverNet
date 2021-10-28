import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import Json_Parser
from covernet.network import CoverNet
from covernet.dataset_covernet import NuSceneDataset_CoverNet
from torch.utils.data.dataloader import DataLoader
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
        self.trajectory_set = pickle.load(open(self.traj_set_path, 'rb'))
        
        self.nuscenes = NuScenes(self.config['DATASET']['dataset_str'], dataroot=self.config['DATASET']['dataset_path'], verbose=verbose)
        self.helper = PredictHelper(self.nuscenes)
        self.dataset = NuSceneDataset_CoverNet(train_mode=dataset_train_mode, config_file_name=config_file, verbose=verbose)
        self.rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=1)    
        self.dataloader = DataLoader(self.dataset, batch_size=dataset_batch_size, shuffle=True)


    def draw_traj_set(self):
        for i in range(len(self.trajectory_set)):
            t = np.array(self.trajectory_set[i][:])
            plt.plot(t[:,0],t[:,1])
        plt.show()


    def draw_on_image(self, img, pred, label, gt, resolution):
        pass


    def run(self):
        # image rasterize
        # eval 결과값 가장 높은 확률값 가지는 인덱스 추출하여 traj_set에서 선택 (pred) / label값 불러와서 traj_set에서 선택 (gt)
        # true lane은 future annotation 불러와서 x,y,yaw로 line customizing
        # pred, gt, true lane visualize cv2.imshow

        self.model.load_state_dict(torch.load(self.model_weight_path))
        print("00")

        for data in self.dataloader:
            self.model.eval()
            print("11")
            img_tensor = data['img'].to(self.device)
            agent_state_tensor = torch.Tensor(data['ego_state'].tolist()).to(self.device)
            agent_state_tensor = torch.squeeze(agent_state_tensor, 1)
            print("22")

            prediction = self.model(img_tensor, agent_state_tensor)
            print("33")

            pred = F.softmax(prediction,dim=-1)

            print("pred :", pred[0].argmax)
            label = data['label']
            print("label :", label[0])




if __name__ == "__main__":
    covernet_test = CoverNet_test(config_file='./covernet/covernet_config.json', 
                                 trained_model_path='./result/model/20211025-20_35_49/epoch_4.pth',
                                 dataset_train_mode=False, 
                                 dataset_batch_size=8,
                                 num_modes=64, 
                                 verbose=False)
    covernet_test.run()
