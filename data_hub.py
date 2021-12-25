import json
import pandas as pd
from utils import get_dataset_with_adjacency_mat
from dhaka_data import DhakaDataCenter

class DataGenerator(object):

    def __init__(self,model_config):
        """ Creates dataset according to the model config
        

        Args:
            path_to_json (str): path to config file 
        """

        self.model_config = model_config
        
        self.data_resampling_method = self.model_config['data_resampling_method']

        self.prediction_length = self.model_config['model_parameters']['prediction_length']

        self.dataset_paths = self.model_config['dataset_paths']['model_data_paths']

        self.train_split = self.model_config['train_test']['train_days']
        self.test_split = self.model_config['train_test']['test_days']

        self.datasets = list()


        print(" self.data_resampling_method: ", self.data_resampling_method)
        print(" self.prediction_length: ", self.prediction_length)
        print(" self.dataset_paths: ",self.dataset_paths)
        print("self.train_split: ",self.train_split)
        print("self.test_split: ",self.test_split)
    
    def build_dataset(self):
        """build the dataset from the prescribed raw data
        path
        """
        print("building datasets....")

        train_data, train_label = list(), list()
        test_data, test_label = list(), list()

        for path in self.dataset_paths:

            print(f"looping in....{path}")
            ds, ds_numpy, adj= get_dataset_with_adjacency_mat(
                path,self.data_resampling_method
            )
            # print(ds.head(10))
            ds.rename(
                columns={
                    'index' : 'timestamp'
                },
                inplace= True
            )

            ds['timestamp'] = ds['timestamp'].apply(lambda x : pd.to_datetime(x))
            ds['day'] = ds['timestamp'].apply(lambda x: x.day)
            ds['hour'] = ds['timestamp'].apply(lambda x : x.hour)
            ds['minute'] = ds['timestamp'].apply(lambda x: x.minute)

            self.datasets.append(ds)
        
        if len(self.datasets) == 1:

            assert (len(self.train_split) == 1 & len(self.test_split) == 1),'error in train_test config'
            print('creating train data and train label...')
            train_data, train_label = DhakaDataCenter(
                dataframe = self.datasets[0],
                pred_len = self.prediction_length,
                date_range = (self.train_split[0][0],self.train_split[0][1])
            ).load_data()
            print('creating test data and test label...')
            test_data, test_label = DhakaDataCenter(
                dataframe = self.datasets[0],
                pred_len = self.prediction_length,
                date_range = ((self.test_split[0][0],self.test_split[0][1]))
            ).load_data()
        else:
            
            for i in range(len(self.train_split)):
                print('creating train data and train label...')
                month_train_data,month_train_label = DhakaDataCenter(
                    dataframe = self.datasets[i],
                    pred_len = self.prediction_length,
                    date_range = (self.train_split[i][0],self.train_split[0][1])
                ).load_data()
                train_data += month_train_data
                train_label += month_train_label

            test_starts = len(self.train_split) 
            for j in range(len(self.test_split)):

                print('creating test data and test label')
                month_test_data,month_test_label = DhakaDataCenter(
                    dataframe = self.datasets[j+test_starts],
                    pred_len = self.prediction_length,
                    date_range = (self.test_split[j][0],self.test_split[j][1])
                ).load_data()
                test_data += month_test_data
                test_label += month_test_label

                



        return train_data, train_label, test_data, test_label, adj   
        



# if __name__ == '__main__':

#     data_generator = DataGenerator('./model_config.json')
#     print(data_generator.train_split[0][0])
#     train_data,train_label,test_data,test_label = data_generator.build_dataset()

#     print(train_data[0].shape)
#     print(test_data[0].shape)
 
