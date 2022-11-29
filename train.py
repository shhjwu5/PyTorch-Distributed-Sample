from servers import *
from clients import *
from models import *
from datasets import *

import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam,SGD
import torch

from torch.utils.data import DataLoader
from datasets import MyDataset
# from attack import attack

devices = [0,1,2,7]

def main():
    rank,world_size = init_dist()

    args = {
        "dataset_type":"random","num_client":world_size-1,"num_shadow":10,
        "rank":rank,"world_size":world_size,"batch_size":32,"local_epoch":1,
        "train_type":"deep","communication_type":"deep","lr":0.01,
        "Model":ExampleNet,"Loss_function":nn.CrossEntropyLoss(),"Optimizer":SGD
    }
    epochs = 100
    
    if rank==world_size-1:
        agent = ServerFedAvg(args)
        agent.train(epochs)
    else:
        args["device"] = torch.device("cuda:%d"%(devices[rank%len(devices)]))
        args["Trainloader"] = DataLoader(MyDataset(file_path="./datasets/%s/"%(args["dataset_type"]),sample_type="train",rank=args["rank"]),batch_size=args["batch_size"],shuffle=True)
        args["Testloader"] = DataLoader(MyDataset(file_path="./datasets/%s/"%(args["dataset_type"]),sample_type="test",rank=args["rank"]),batch_size=args["batch_size"],shuffle=True)
        agent = ClientFedAvg(args)
        agent.train(epochs)

        torch.save(agent.model,"./shadow_models/model_%d.pt"%(rank))

        # if rank==0:
        #     attack(agent.model,args)

def init_dist():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return [rank, world_size]

if __name__=="__main__":
    main()