## Execute tensorboard => tensorboard --logdir=./result/tensorboard
import os
import sys

from torch.utils import data
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from datetime import datetime
from utils import Json_Parser
from network import CoverNet

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
        self.criterion = nn.CrossEntropyLoss()           ## classification loss
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

                
    def run(self):
        print("CoverNet learning starts!")
        step = 1
        best_val_loss = 10000
        for epoch in range(self.n_epochs + 1):
            Loss, Val_Loss = [], []

            for data in self.train_dataset:
                # train_mode
                self.model.train()

                img_tensor = data['img'].to(self.device)
                agent_state_tensor = torch.Tensor(data['ego_state'].tolist()).to(self.device)
                agent_state_tensor = torch.squeeze(agent_state_tensor, 1)

                prediction = self.model(img_tensor, agent_state_tensor)
                pred = F.softmax(prediction,dim=-1)
                label = data['label']

                self.optimizer.zero_grad()
                loss = self.criterion(pred,label)

                ## for calculating gt loss
                label_onehot = F.one_hot(label, num_classes=self.num_modes)
                gt_loss = self.criterion(label_onehot.float(),label)
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
