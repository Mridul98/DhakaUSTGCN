import torch.nn as nn


class Regression(nn.Module):

    def __init__(self, emb_size, out_size):
        super(Regression, self).__init__()

        self.layer = nn.Sequential(nn.Linear(emb_size, emb_size),
                               
                                nn.Tanhshrink(),
                                nn.Linear(emb_size, emb_size),
                                nn.Tanhshrink(),
                                nn.Linear(emb_size, out_size),
                                nn.ReLU(),
                                
                                )
            

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embds):
        # print('embds shape: ',embds.shape)
        logists = self.layer(embds)
        # print('logists shape: ',logists.shape)
        return logists