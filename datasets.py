import numpy as np
from torch.utils.data import Dataset
import torch

import numpy as np
import sys
sys.path.append('/root/Privacy_copy/attacks/ml_privacy_meter')

# from privacy_meter.dataset import Dataset as PrivacyDataset

class MyDataset(Dataset):
    def __init__(self,file_path="./datasets/single/",rank=None,sample_type="train",args=None):
        if rank is None:
            self.x = torch.tensor(np.concatenate([np.load(file_path+str(k)+"/"+sample_type+"_x.npy") for k in range(args["num_client"])],axis=0),dtype=torch.float)
            self.y = torch.tensor(np.concatenate([np.load(file_path+str(k)+"/"+sample_type+"_y.npy") for k in range(args["num_client"])],axis=0),dtype=torch.float)
        else:
            self.x = torch.tensor(np.load(file_path+str(rank)+"/"+sample_type+"_x.npy"),dtype=torch.float)
            self.y = torch.tensor(np.load(file_path+str(rank)+"/"+sample_type+"_y.npy"),dtype=torch.float)
        print(rank,sample_type,len(self))
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index],self.y[index]

