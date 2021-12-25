
import torch
import pandas as pd
import numpy as np
import random
from datetime import datetime
from data_hub import DataGenerator
from traffic_model import TrafficModel
from utils import get_model_config


def produce_pred_file_of_day(day_date,road_number,pred,label):
        
    # result generation
    prediction_15 = pd.DataFrame(data = {
        'timestamp' : pd.date_range(datetime(year =2020,hour = 7 ,day = day_date,month = 1, minute = 0),periods = 204,freq='5min')
    })
    road_number = road_number
    pred_list = list()
    label_list = list()
    for i in range(road_number,len(label),154):
        pred_list.append(pred[i])
        label_list.append(label[i])
    road_dict = dict()
    list_pred = list()
    list_lab = list()
    for i in range(0,len(pred_list),3):  #change here
        for j in range(len(pred_list[i])):
            list_pred.append(pred_list[i][j])
            list_lab.append(label_list[i][j])

    road_dict[road_number] = list_pred
    road_dict['l'] = list_lab
    prediction_15[str(road_number)] = road_dict[road_number]
    prediction_15[str(road_number)+'l'] = road_dict['l']
    # prediction_15[str(road_number)+'l'] = scalers[road_number].inverse_transform(prediction_15.iloc[:204][str(road_number)+'l'].values.reshape((-1,1)))
    # prediction_15[str(road_number)] = scalers[road_number].inverse_transform(prediction_15.iloc[:204][str(road_number)].values.reshape((-1,1)))
    prediction_15.iloc[:204].plot(x ='timestamp',y =[str(road_number),str(road_number)+'l'])
    return prediction_15.iloc[:204].set_index('timestamp')

def produce_all(day_date,pred,label,filename):
    df_list = list()
    final_df = None
    for i in range(0,154):
        df_list.append(produce_pred_file_of_day(day_date,i,pred,label))
    final_df =  pd.concat(df_list,axis = 1)
    final_df.to_csv(f"results/{filename}_{day_date}.csv")

    return final_df



if __name__ == '__main__':

   
    model_config = get_model_config('./model_config.json')
    
    model_params = model_config['model_parameters']
    
    is_cuda = model_params['cuda']
    seed  = model_params['seed']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tier_df = model_config['dataset_paths']['tier_info_path']
    tier_info = pd.read_excel(tier_df)
    tier_list = tier_info[model_config['dataset_paths']['tier_column_name']].values.tolist()

    pred_len = model_params['prediction_length']
    num_timestamps = model_params['number_of_timestamps']
    GNN_layers = model_params['gnn_layers']
    input_size = model_params['input_size']
    out_size = model_params['output_size']
    epochs = model_params['epoch']
    device = torch.device("cuda" if is_cuda else "cpu")
    is_trained_model = model_params['trained_model']
   

    pred = []
    label = []
    
    data_generator = DataGenerator(model_config)
    print(data_generator.train_split[0][0])
    train_data,train_label,test_data,test_label,adj = data_generator.build_dataset()

    print(train_data[0].shape)
    print(test_data[0].shape)


    traffic_model = TrafficModel(train_data = train_data,train_label = train_label,
                                test_data = test_data,test_label = test_label,
                                adj = adj,input_size=input_size,out_size=out_size,
                                GNN_layers=GNN_layers,epochs=epochs,device=device,
                                num_timestamps=num_timestamps,pred_len=pred_len,
                                save_flag=True,t_debug=False,b_debug=False,
                                tier_list=tier_list
                                )

    if is_trained_model is False:

        traffic_model.run_model()
    else:
        print('running_trained_model')
        pred,label,idx = traffic_model.run_Trained_Model()

        # for result generation
        produce_all(14,pred,label,'january_normal_pred_after_15_14')

       

 