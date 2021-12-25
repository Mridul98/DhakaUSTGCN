import torch
import numpy as np
import math


#apply_model
def apply_model(train_nodes, CombinedGNN, regression, 
                node_batch_sz, device,train_data,
                train_label,avg_loss,lr,pred_len):
    

    # assert loss_weights.is_cuda
    models = [CombinedGNN, regression]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)



    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)

    optimizer.zero_grad()  # set gradients in zero...
    for model in models:
        model.zero_grad()  # set gradients in zero

    node_batches = math.ceil(len(train_nodes) / node_batch_sz)

    loss = torch.tensor(0.).to(device)
    #window slide
    raw_features = train_data
    labels = train_label
    for index in range(node_batches):
    
        nodes_batch = train_nodes[index * node_batch_sz:(index + 1) * node_batch_sz]
        nodes_batch = nodes_batch.view(nodes_batch.shape[0],1)
        labels_batch = labels[nodes_batch]      
        labels_batch = labels_batch.view(len(labels_batch),pred_len)
        embs_batch = CombinedGNN(raw_features)  # Find embeddings for all the ndoes in nodes_batch
        
        logists = regression(embs_batch)

        #print('label shape: ',labels_batch.shape)
        #print('predicted shape: ', logists.shape)
        
        loss_sup = torch.nn.MSELoss()(logists, labels_batch)
        #print(loss_sup.shape)

        loss_sup /= len(nodes_batch)
        loss += loss_sup



    avg_loss += loss.item()

    loss.backward()
    # for model in models:
    #   nn.utils.clip_grad_norm_(model.parameters(), 5) 
    optimizer.step()

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    return CombinedGNN, regression,avg_loss

def evaluate(test_nodes,raw_features,
             labels, graphSage,regression,
             device,test_loss):


    models = [graphSage, regression]

    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    
    val_nodes = test_nodes
    embs = graphSage(raw_features)
   
    predicts = regression(embs)
    loss_sup = torch.nn.MSELoss()(predicts, labels)
    loss_sup /= len(val_nodes)
    test_loss += loss_sup.item()

    for param in params:
        param.requires_grad = True

    return predicts,test_loss

def RMSELoss(yhat,y):
    yhat = torch.FloatTensor(yhat)
    y = torch.FloatTensor(y)
    return torch.sqrt(torch.mean((yhat-y)**2)).item()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
