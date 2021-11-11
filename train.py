## Execute tensorboard => tensorboard --logdir=./result/tensorboard
import os
import pickle
import sys
from matplotlib import pyplot as plt
from nuscenes.prediction.models.covernet import ConstantLatticeLoss

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import numpy as np
import torch

from datetime import datetime
from utils import Json_Parser
from network import CoverNet
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from dataset_covernet import NuSceneDataset_CoverNet
from nuscenes.prediction.models.backbone import ResNetBackbone
from torch.utils.tensorboard import SummaryWriter

class CoverNet_train:
    def __init__(self, config_file, verbose):
        self.parser = Json_Parser(config_file)
        self.config = self.parser.load_parser()    
        self.device = torch.device(self.config['LEARNING']['device'] if torch.cuda.is_available() else 'cpu')
        self.lr = self.config['LEARNING']['lr']
        self.momentum = self.config['LEARNING']['momentum']
        self.n_epochs = self.config['LEARNING']['n_epochs']
        self.batch_size = self.config['LEARNING']['batch_size']
        self.val_batch_size = self.config['LEARNING']['val_batch_size']
        self.num_val_data = self.config['LEARNING']['num_val_data']
        self.num_modes = self.config['LEARNING']['num_modes']
        self.print_size = self.config['LEARNING']['print_size']

        self.train_dataset = DataLoader(NuSceneDataset_CoverNet(train_mode=True, config_file_name=config_file, verbose=verbose), batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_dataset = DataLoader(NuSceneDataset_CoverNet(train_mode=False, config_file_name=config_file, verbose=verbose), batch_size=self.val_batch_size, shuffle=True, num_workers=4)

        self.backbone = ResNetBackbone('resnet50')
        # self.resnet_path = self.config['LEARNING']['weight_path']
        # self.backbone.load_state_dict(torch.load(self.resnet_path))
        self.model = CoverNet(self.backbone, self.num_modes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum) 
        
        ###############################################################
        # self.criterion = nn.CrossEntropyLoss()           ## classification loss
        self.traj_set_path = self.config['LEARNING']['trajectory_set_path']
        self.trajectories_set =torch.Tensor(pickle.load(open(self.traj_set_path, 'rb')))
        self.criterion = ConstantLatticeLoss(self.trajectories_set)
        ###############################################################

        self.model = self.model.to(self.device)
        self.save_name = datetime.now().strftime("%Y%m%d-%H_%M_%S")
        self.writer = SummaryWriter('./result/tensorboard/' + self.save_name)
        self.net_save_path = os.path.join(self.config['LEARNING']['model_save_path'], self.save_name)
        if not os.path.exists(self.net_save_path):
            os.mkdir(self.net_save_path)
        self.writer.add_text('Config', json.dumps(self.config))

        dataset_info = {'train_size' : self.train_dataset.__len__(), 'val_size' : self.val_dataset.__len__(), 
                        'train_batch_size' : self.batch_size, 'val_batch_size' : self.val_batch_size}
        self.writer.add_text('Dataset_size', json.dumps(dataset_info))

    def get_label(self, traj, future):
        scores = torch.full((len(traj),),1e4)
        for i in range(len(traj)):
            if (torch.norm(traj[i,-1]-future[-1]) < 10):    
                scores[i]= torch.norm(traj[i]-future)
            
        ind=torch.argmin(scores)
        
        res=torch.zeros_like(scores)
        res[ind] =1

        return res, ind

    def plot_results(self, data, pred, anchor_ind):
        pred_traj_idx = pred.argmax().detach().cpu().numpy()

        xs = []
        ys = []
        for j in range(len(data['future_local_ego_pos'].detach().cpu().numpy())):
            xs.append(data['future_local_ego_pos'][j][0].detach().cpu().numpy())
            ys.append(data['future_local_ego_pos'][j][1].detach().cpu().numpy())
        xs = np.array(xs)
        ys = np.array(ys)

        xss = []
        yss = []
        label = self.trajectories_set[anchor_ind.detach().cpu().numpy()]
        for j in range(len(label)):
            xss.append(label[j][0].detach().cpu().numpy())
            yss.append(label[j][1].detach().cpu().numpy())
        xss = np.array(xss)
        yss = np.array(yss)

        xsss = []
        ysss = []
        label = self.trajectories_set[pred_traj_idx]
        for j in range(len(label)):
            xsss.append(label[j][0].detach().cpu().numpy())
            ysss.append(label[j][1].detach().cpu().numpy())
        xsss = np.array(xsss)
        ysss = np.array(ysss)


        fig, ax = plt.subplots(1,4, figsize = (10,10))
        # Rasterized Image
        ax[0].imshow((data['img'].squeeze(0).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
        ax[0].set_title("Rasterized Image")
        # Real ego future history
        ax[1].set_title("Ego future history")
        ax[1].plot(xs, ys, 'bo')
        ax[1].set_aspect('equal')
        ax[1].set_xlim(-30, 30)
        ax[1].set_ylim(-10, 50)
        # Label of traj_set
        ax[2].plot(xss,yss,'yo')
        ax[2].set_aspect('equal')
        ax[2].set_xlim(-30,30)
        ax[2].set_ylim(-10,50)
        ax[2].set_title("{}th anchor".format(data['label'].detach().cpu().numpy()))
        # prediction
        ax[3].plot(xsss,ysss,'yo')
        ax[3].set_aspect('equal')
        ax[3].set_xlim(-30,30)
        ax[3].set_ylim(-10,50)
        ax[3].set_title("Prediction")
                
        return fig 


    def run(self):
        print("CoverNet learning starts!")
        step = 1
        best_val_loss = 10000
        for epoch in range(self.n_epochs + 1):
            Loss, Val_Loss = [], []

            for data in self.train_dataset:
                # train_mode
                self.model.train()

                img_tensor = data['img'].to(device=self.device)
                agent_state_tensor = torch.Tensor(data['ego_state'].tolist()).to(self.device)
                agent_state_tensor = torch.squeeze(agent_state_tensor, 1)

                prediction = self.model(img_tensor, agent_state_tensor)
                # label = data['label']
                label, anchor_ind = self.get_label(self.trajectories_set, data['future_local_ego_pos'])

                self.optimizer.zero_grad()
                loss = self.criterion(prediction,label)

                ## for calculating gt loss
                # label_onehot = F.one_hot(label, num_classes=self.num_modes)
                # gt_loss = self.criterion(label_onehot.float(),label)
                # print("gt_loss : ",gt_loss)

                loss.backward()
                self.optimizer.step()

                step += 1
                with torch.no_grad():
                    Loss.append(loss.cpu().detach().numpy())

                if step % self.print_size == 0:
                    with torch.no_grad():
                        # eval_mode
                        self.model.eval()

                        k = 0
                        for val_data in self.val_dataset:
                            img_tensor = val_data['img'].to(self.device)
                            agent_state_tensor = torch.Tensor(val_data['ego_state'].tolist()).to(self.device)
                            agent_state_tensor = torch.squeeze(agent_state_tensor, 1)

                            prediction = self.model(img_tensor, agent_state_tensor)
                            label = val_data['label']

                            val_loss = self.criterion(prediction,label)
                            Val_Loss.append(val_loss.detach().cpu().numpy())

                            pred = F.softmax(prediction,dim=-1)
                            self.writer.add_figure('Results', self.plot_results(val_data,pred,anchor_ind), step)
                            k += 1
                            if(k == self.num_val_data):
                                break
                            
                    loss = np.array(Loss).mean()
                    val_loss = np.array(Val_Loss).mean()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(self.net_save_path, 'best_val_loss_model.pth')
                        torch.save(self.model.state_dict(), save_path)
                        self.writer.add_scalar('Best Val Loss', best_val_loss)

                    self.writer.add_scalar('Loss', loss, step)
                    self.writer.add_scalar('Val Loss', val_loss, step)


                    print("Epoch: {}/{} | Step: {} | Loss: {:.5f} | Val_Loss: {:.5f}".format(
                            epoch + 1, self.n_epochs, step, loss, val_loss))                    
                    Loss, Val_Loss = [], []
        
            save_path = os.path.join(self.net_save_path, 'epoch_{0}.pth'.format(epoch + 1))
            torch.save(self.model.state_dict(), save_path)

        save_path = os.path.join(self.net_save_path, 'CoverNet.pth')
        torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    covernet_train = CoverNet_train(config_file='./covernet_config.json', verbose=False)
    covernet_train.run()
