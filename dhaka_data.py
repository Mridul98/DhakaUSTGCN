import pandas as pd
import numpy as np


class DhakaDataCenter(object):

    def __init__(self,dataframe,pred_len,date_range):

        self.pred_len = pred_len
        self.date_range = date_range
        self.dataframe = dataframe

        self.TOT_NODE = 154
        self.DAY = 8

    
    def load_one_day_data(self,day):
        """load traffic data(all roads) of all the 18 hours 
        where each hours has 12 samples of a day
        (from 6 am to 12 pm) 

        Args:
            day (int): day index
        """
        ts_data = list()
        label_data = list()
        a = list()
        for i in range(18):

            for j in range(0,56,5):

                df , df_numpy = self.fetch_data(6+i,j,[day])

                #print(df)
                #print(df_numpy.shape)
    
    def fetch_data(self,hour,minute,days):
        """fetch data of given hour , minute and days indexes of a month

        Args:
            hour (int): hour
            minute (int): minute
            days (int): second
        """
        # print("hour: ",hour)
        # print("minute: ", minute)
        # print("days: ",days)
        cond =(
            (self.dataframe['hour'] == hour) 
            & (self.dataframe['minute'] == minute) 
            & (self.dataframe['day'].isin(days))
        )
        
        
        df = self.dataframe[cond][self.dataframe.columns.values[0:-3]]
        
        numpy_df = np.asarray(
            self.dataframe[cond][self.dataframe.columns.values[1:-3]].T
        ).reshape(1,self.TOT_NODE,self.DAY)
        
        return df ,numpy_df

    
    def load_data(self):

        """Loads final dataset according to the day settings
        """
        ts_data = list()
        label_data = list()

        # storing train/test days indexes in a list
        train_days = [i for i in range(self.date_range[0],self.date_range[1])]
        
        for i in range(len(train_days)):
            # we need 8 days data for prediction.
            # So,the condition is checking 
            # whether we are 
            # taking the exact 8 days or not
            if(len(train_days[i:i+8])<8):
                break

            days = train_days[i:i+8] #sliding over days
            print('-------')
            print(days)
            a = list()

            # same as the function named 'load_one_day_data'
            # but it is only for train/test 
            # data processing purposes
            for i in range(18):   # iterating over hours

                for j in range(0,56,5):  # iterating over each timestamp of each hour

                    df, df_numpy = self.fetch_data(6+i,j,days) #fetching given days data 

                    #print(df)
                    #print(df_numpy)

                    a.append(df_numpy) #appending to a list 

            a_data = list()

            pred_len = self.pred_len # prediction length
            
            print('''-------processinng feature data-------''')
            for i in range(len(a)-(pred_len-1)):

                train_st = i 
                train_en = i+12

                if train_en > (len(a)-(pred_len)):
                    break

                ts_data.append(
                    np.concatenate(a[train_st:train_en],axis = 0)
                )



            print('''-----------processing label_data----------''')
            print('pred_len: ',pred_len)
            
            for j in range(12,len(a)):

                label_st = j
                label_end = j + pred_len

                if label_end > len(a):
                    break

                labels = list()

                for k in range(label_st,label_end):

                    labels.append(a[k][:,:,-1].reshape(self.TOT_NODE,1))

                labels = np.concatenate(labels,axis = 1)

                #print(j,labels)

                label_data.append(labels)
        
        return ts_data , label_data



