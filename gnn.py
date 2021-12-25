import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import filterfalse
from sklearn.cluster import KMeans

class SPTempGNN(nn.Module):
    
    def __init__(self,D_temporal,A_temporal,num_timestamps,out_size,tot_nodes,tier_list):
        super(SPTempGNN, self).__init__()
        # self.A_neighbor = A_neighbor
        # self.D_neighbor = D_neighbor
        self.tot_nodes = tot_nodes
        self.sp_temp = torch.mm(D_temporal,torch.mm(A_temporal,D_temporal))
        self.tier_one_count = 76
        self.tier_two_count = 78
        self.tier_list = tier_list
        #### previous implementation
        # self.his_temporal_weight_tier_one = nn.Parameter(torch.FloatTensor(num_timestamps,out_size))
        # self.his_temporal_weight_tier_two = nn.Parameter(torch.FloatTensor(num_timestamps,out_size))
        #### previous implementation finished
        self.his_temporal_weight = nn.Parameter(torch.FloatTensor(num_timestamps,out_size)) #(12,8)

        #### alternative implementation
        self.his_temporal_weight_tier_one = nn.Parameter(torch.FloatTensor(1,out_size))
        self.his_temporal_weight_tier_two = nn.Parameter(torch.FloatTensor(1,out_size))
        self.his_temporal_weight_tier_three = nn.Parameter(torch.FloatTensor(1,out_size))
        self.his_temporal_weight_tier_four = nn.Parameter(torch.FloatTensor(1,out_size))

        self.mixed_weights = self.create_mixed_weights()
        #### alternative implementation finished

        self.diff_his_temporal_weight = nn.Parameter(torch.FloatTensor(num_timestamps*self.tot_nodes,out_size))
        print('shape of his_temporal_weight: ',self.his_temporal_weight.shape)
        # self.cur_temporal_weight = nn.Parameter(torch.FloatTensor(num_timestamps,1))
        # self.init_params()

        self.his_final_weight_self = nn.Parameter(torch.FloatTensor(out_size,out_size//2))
        self.his_final_weight_sptemp = nn.Parameter(torch.FloatTensor(out_size,out_size//2))
        # self.cur_final_weight = nn.Parameter(torch.FloatTensor(2*1,1))

        ### for tier conditioning, shape of self.his_final_weight will be (out_size,out_size)
        self.his_final_weight = nn.Parameter(torch.FloatTensor(out_size,out_size))
        # self.init_params()

    def create_mixed_weights(self):
        tensor_list = list()
        for i in self.tier_list:
            if i == 0:
                tensor_list.append(self.his_temporal_weight_tier_one)
            if i == 1:
                tensor_list.append(self.his_temporal_weight_tier_two)
            if i == 2:
                tensor_list.append(self.his_temporal_weight_tier_three)
            if i == 3:
                tensor_list.append(self.his_temporal_weight_tier_four)
                
        final_tensor = torch.cat(tensor_list,0)
        return final_tensor

    def init_params(self):
        nn.init.normal_(self.his_temporal_weight_tier_one,mean=0.5,std=1.5)
        nn.init.normal_(self.his_temporal_weight_tier_two,mean=5.5,std=1.5)
        
  
    def forward(self,his_raw_features):#,cur_raw_features):
        his_self = his_raw_features
        # print('his_raw_features shape :',his_raw_features.shape )
        # print('his_temporal_weights : ',self.his_temporal_weight.shape)



        # ### uncomment next line when using tier conditioning
        # his_temporal = self.create_mixed_weights().repeat(12,1) * his_raw_features

        # his_temporal = torch.mm(self.sp_temp,his_temporal)

        # his_self_aggregated = his_self.mm(self.his_final_weight_self)

        # his_temporal_aggregated = his_temporal.mm(self.his_final_weight_sptemp)

        # his_combined = torch.cat([his_self_aggregated,his_temporal_aggregated], dim=1)
        # #######



        ### this section is for normal ustgcn
        his_temporal = self.his_temporal_weight.repeat(self.tot_nodes,1) * his_raw_features
        his_temporal = torch.mm(self.sp_temp,his_temporal)
        his_combined = torch.cat([his_self,his_temporal], dim=1)
        ####
        
        his_raw_features =F.relu(his_combined.mm(self.his_final_weight))

    

        
        return his_raw_features#,cur_raw_features



#Combined graphsage

class CombinedGNN(nn.Module):

    def __init__(self,input_size,out_size, adj_lists,
                 device,st,GNN_layers,num_timestamps,
                 day,tier_list):
                 
        super(CombinedGNN, self).__init__()

        self.st = st
        self.num_timestamps = num_timestamps
        self.out_size = out_size
        self.tot_nodes = adj_lists.shape[0]
        self.device = device
        self.adj_lists = adj_lists 
        self.GNN_layers = GNN_layers

        self.day = day
        self.tier_list = tier_list
       
        # TODO creates weight matrices to aggregrate
        # roads within same cluster with same matrix 
        
                
        self.his_weight = nn.Parameter(torch.FloatTensor(out_size, self.num_timestamps*out_size))
        self.cur_weight = nn.Parameter(torch.FloatTensor(1, self.num_timestamps*1))
        
        A = self.adj_lists
        dim = self.num_timestamps*self.tot_nodes

        A_temporal = torch.zeros(dim,dim)
        D_temporal = torch.zeros(dim,dim)
        identity = torch.eye(self.tot_nodes)
        A=A.to(self.device)
        A_temporal=A_temporal.to(self.device)
        identity=identity.to(self.device)
        D_temporal=D_temporal.to(self.device)
        for i in range(0, self.num_timestamps):
            for j in range(0, i+1):

                row_st = i*self.tot_nodes
                row_en = row_st + self.tot_nodes
                col_st = j*self.tot_nodes
                col_en = col_st + self.tot_nodes

                if i == j: #adj matrix  
                    A_temporal[row_st:row_en,col_st:col_en] = A 
                else: #identity matrix
                    A_temporal[row_st:row_en,col_st:col_en] = identity + A
            
        row_sum = torch.sum(A_temporal,0)

        for i in range(dim):
            D_temporal[i,i] = 1/max(torch.sqrt(row_sum[i]),1)      
        
        for i in range(GNN_layers):
            sp_temp = SPTempGNN(D_temporal,A_temporal,
                                num_timestamps,out_size,
                                self.tot_nodes,self.tier_list)
            setattr(self, 'sp_temp_layer'+str(i), sp_temp)
          

        dim2 = self.num_timestamps*(out_size)
        self.final_weight = nn.Parameter(torch.FloatTensor(dim2, dim2))

       
        self.init_params()
       
        
        

        

    def init_params(self):
      
      # for name, param in self.named_parameters():
      #   if param.requires_grad:
      #     # print name, param.data
      #     print(name)


        for param in self.parameters():
            if(len(param.shape)>1):
                nn.init.xavier_uniform_(param)
      
      # nn.init.normal_(self.his_final_weight_tier_one,mean=0.5,std=1)
      # nn.init.normal_(self.his_final_weight_tier_two,mean=0,std=1)


    def apply_kmeans(self,np_array,n_clusters):

        cluster_index = [i for i in range(n_clusters)]
        cluster_members = dict()

        print('starting....')
        print('applying k_means....')
        k_means = KMeans(n_clusters=n_clusters,random_state=824,).fit(np_array)
        # print(k_means.labels_)
        labels = enumerate(k_means.labels_)
        for i in cluster_index:

            cluster_members[str(i)] = list(

                filterfalse(
                    lambda x : x[1] != cluster_index[i],
                    labels
                )
            )
        
        return cluster_members
    def create_weight_matrices(self,n_clusters,out_size):

        for i in range(n_clusters):

            setattr(

                self,f'weight_cluster_{i}',
                nn.Parameter(
                    torch.FloatTensor((96,out_size))
                )
            )


    def forward(self,his_raw_features):
      
        dim = self.num_timestamps*self.tot_nodes
        his_raw_features = his_raw_features[:,:,:self.day].view(dim,self.day)

      
        for i in range(self.GNN_layers):
            sp_temp = getattr(self, 'sp_temp_layer'+str(i))
            his_raw_features = sp_temp(his_raw_features)
        # his_raw_features = sp_temp(his_raw_features)
      
     
      
        his_list = []
      # cur_list = []

        for i,timestamp in enumerate(range(self.num_timestamps)):
            st = timestamp * self.tot_nodes
            en = (timestamp+1) * self.tot_nodes
            # print(f'his_raw_features shape: {his_raw_features.shape}')
            his_list.append(his_raw_features[st:en,:])
            # cur_list.append(cur_raw_features[st:en,:])
        # print(f'his list length : {len(his_list)}')
        # print('component list: ',his_list[0].shape)
        his_final_embds = torch.cat(his_list,dim=1)
      # cur_final_embds = torch.cat(cur_list,dim=1)
        
        final_embds = [his_final_embds]
        final_embds = torch.cat(final_embds,dim=1)
        final_embds = F.relu(self.final_weight.mm(final_embds.t()).t())
        
 

        return final_embds #154x96



