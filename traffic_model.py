import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from regression import Regression
from gnn import (
    SPTempGNN,
    CombinedGNN
)
from model_utils import (
    apply_model,
    evaluate,
    RMSELoss,
    mean_absolute_percentage_error,
)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(filename_suffix='ustgcn_15_origin')
RMSES = list()

class TrafficModel(object):

    def __init__(self, train_data,train_label,
                 test_data,test_label,adj, 
                 input_size, out_size,GNN_layers,
                 epochs, device,num_timestamps, 
                 pred_len,save_flag,
                 t_debug,b_debug,tier_list):
      
        super(TrafficModel, self).__init__()
        
        
        self.train_data,self.train_label,self.test_data,self.test_label,self.adj = train_data,train_label,test_data,test_label,adj
        self.all_nodes = [i for i in range(self.adj.shape[0])]

        #self.ds = ds
        self.input_size = input_size
        self.out_size = out_size
        self.GNN_layers = GNN_layers
        self.day = input_size 
        self.device = device
        self.epochs = epochs
    
        self.regression = Regression(input_size * num_timestamps, pred_len)
        self.num_timestamps = num_timestamps
        self.pred_len = pred_len
        self.tier_list = tier_list

        self.node_bsz = 512
        #self.PATH = PATH
        self.save_flag = save_flag

        self.train_data = torch.FloatTensor(self.train_data).to(device)
        self.test_data = torch.FloatTensor(self.test_data).to(device)
        self.train_label = torch.FloatTensor(self.train_label).to(device)
        self.test_label = torch.FloatTensor(self.test_label).to(device)
        self.all_nodes = torch.LongTensor(self.all_nodes).to(device)
        self.adj = torch.FloatTensor(self.adj).to(device)
       
        self.t_debug = t_debug
        self.b_debug = b_debug

      
    def run_model(self):


        timeStampModel = CombinedGNN(self.input_size,self.out_size,
                                    self.adj,self.device,1,
                                    self.GNN_layers,self.num_timestamps,
                                    self.day,self.tier_list)

        timeStampModel.to(self.device)
    
        regression = self.regression
        regression.to(self.device)

        min_RMSE = float("Inf") 
        min_MAE = float("Inf") 
        min_MAPE = float("Inf")
        best_test = float("Inf")
        

        
        lr = 0.0001
            
        train_loss = torch.tensor(0.).to(self.device)  
        for epoch in range(1,self.epochs):

            print("Epoch: ",epoch," running...")

            tot_timestamp = len(self.train_data)
            if self.t_debug:
                tot_timestamp = 60
            idx = np.random.permutation(tot_timestamp)

            for data_timestamp in idx:

                tr_data = self.train_data[data_timestamp]
                tr_label = self.train_label[data_timestamp]

                timeStampModel, regression, train_loss = apply_model(self.all_nodes,timeStampModel, 
                                                                regression,self.node_bsz, 
                                                                self.device,tr_data,
                                                                tr_label,train_loss,lr,self.pred_len)

                if self.b_debug:
                    break

            train_loss /= len(idx)
            # if epoch == 24 and epoch%8==0:
            #   lr *= 0.5
            # else:
            #   lr = 0.00001
            # if epoch == 31:
            #   lr *= 0.5

            # if epoch == 4: 
            #     lr *= 0.5

            print("Train avg loss: ",train_loss)


            pred = []
            label = []
            tot_timestamp = len(self.test_data)

            if self.t_debug:
                tot_timestamp = 60

            idx = np.random.permutation(tot_timestamp)
            test_loss = torch.tensor(0.).to(self.device)
            for data_timestamp in idx:

          
          
            #test_label
                raw_features = self.test_data[data_timestamp]
                test_label = self.test_label[data_timestamp]
            
            #evaluate
                temp_predicts,test_loss = evaluate(self.all_nodes,raw_features,
                                                test_label,timeStampModel,
                                                regression, self.device,test_loss)

                label = label + test_label.detach().tolist()
                pred = pred + temp_predicts.detach().tolist()

            

                if self.b_debug:
                    break

        
            test_loss /= len(idx)
            print("Average Test Loss: ",test_loss)
            writer.add_scalar('validation loss : ',test_loss,epoch)
            if test_loss <= best_test:
                best_test = test_loss
                pred_after = self.pred_len * 5
                if self.save_flag:
                    torch.save(timeStampModel, f"./test_jan_ustgcn_normal/bestTmodel_{pred_after}minutes_GNN_4_0001.pth" )
                    torch.save(regression,f"./test_jan_ustgcn_normal/bestRegression_{pred_after}minutes_GNN_4_0001.pth")

            RMSE = torch.nn.MSELoss()(torch.FloatTensor(pred), torch.FloatTensor(label))
            RMSE = torch.sqrt(RMSE).item()
            MAE = mean_absolute_error(pred,label)
            MAPE = mean_absolute_percentage_error(label,pred)
        
            RMSES.append(RMSE)
        
            print("Epoch:", epoch)
            print("RMSE: ", RMSE)
            print("MAE: ", MAE)
            print("MAPE: ", MAPE)
            print("===============================================")

            min_RMSE = min(min_RMSE,RMSE)
            min_MAE = min(min_MAE,MAE)
            min_MAPE = min(min_MAPE,MAPE)
        
            print("Min RMSE: ", min_RMSE)
            print("Min MAE: ", min_MAE)
            print("Min MAPE: ", min_MAPE)
        
            print("===============================================")



        return
    
    def run_Trained_Model(self):
        pred_after = self.pred_len * 5
        timeStampModel = torch.load( f"./test_jan_ustgcn_normal/bestTmodel_{pred_after}minutes_GNN_4_0001.pth")
        regression = torch.load( f"./test_jan_ustgcn_normal/bestRegression_{pred_after}minutes_GNN_4_0001.pth")
        pred = []
        label = []
        tot_timestamp = len(self.test_data)
        #idx = np.random.permutation(tot_timestamp+1-self.num_timestamps)
        #idx = np.random.permutation(tot_timestamp)
        idx = np.arange(tot_timestamp)
        test_loss = torch.tensor(0.).to(self.device)
        for data_timestamp in idx:

            #window slide
            #timeStampModel.st = data_timestamp

            #test_label
            raw_features = self.test_data[data_timestamp]
            test_label = self.test_label[data_timestamp]
            
            #evaluate
            temp_predicts,test_loss = evaluate(self.all_nodes,raw_features,
                                            test_label, timeStampModel, 
                                            regression,self.device,
                                            test_loss)

            label = label + test_label.detach().tolist()
            pred = pred + temp_predicts.detach().tolist()
            # print(label.shape)

      
        test_loss /= len(idx)
        print("Average Test Loss: ",test_loss)

        
        RMSE = torch.nn.MSELoss()(torch.FloatTensor(pred), torch.FloatTensor(label))
        RMSE = torch.sqrt(RMSE).item()
        MAE = mean_absolute_error(pred,label)
        MAPE = mean_absolute_percentage_error(label,pred)
        
        
        print("RMSE: ", RMSE)
        print("MAE: ", MAE)
        print("MAPE: ", MAPE)
        print("===============================================")

        return pred,label,idx
