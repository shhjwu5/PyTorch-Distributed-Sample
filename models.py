import torch.nn as nn
import torch
import torch.nn.functional as F

class Multiply(nn.Module):
    def __init__(self):
        super(Multiply,self).__init__()
        self.weight = nn.Parameter(torch.rand((1,10)),requires_grad=True)

    def forward(self,x):
        x.mul_(self.weight)
        return x

class ExampleNet(nn.Module):
    def __init__(self,model_type="deep",comm_type="deep"):
        super(ExampleNet,self).__init__()
        self.model_type = model_type
        self.comm_type = comm_type
        
        if self.model_type=="shallow":
            self.fe1 = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU()
            )
            self.flatten1 = nn.Flatten()
            self.pd1 = nn.Linear(in_features=10*12*12,out_features=10)
            self.softmax = nn.Softmax(dim=1)

            if self.comm_type == "shallow":
                self.communication_layers = [self.fe1]
            elif self.comm_type == "deep":
                self.communication_layers = [self.fe1,self.pd1]

        elif self.model_type=="deep":
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            #self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

            # self.fe1 = nn.Sequential(
            #     nn.Conv2d(1, 10, kernel_size=5),
            #     nn.MaxPool2d(2),
            #     nn.ReLU()
            # )
            # self.fe2 = nn.Sequential(
            #     nn.Conv2d(10, 20, kernel_size=5),
            #     nn.MaxPool2d(2),
            #     nn.ReLU()
            # )
            # self.dropout = nn.Dropout()
            # self.flatten2 = nn.Flatten()
            # self.pd2 = nn.Linear(in_features=20*4*4, out_features=10)
            # self.softmax = nn.Softmax(dim=1)

            if self.comm_type == "shallow":
                self.communication_layers = [self.conv1]
            elif self.comm_type == "deep":
                self.communication_layers = [self.conv1,self.conv2,self.fc1,self.fc2]
        
        elif self.model_type=="mix":
            self.fe1 = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU()
            )
            self.fe2 = nn.Sequential(
                nn.Conv2d(10, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU()
            )
            self.flatten1 = nn.Flatten()
            self.flatten2 = nn.Flatten()
            self.pd1 = nn.Linear(in_features=10*12*12,out_features=10)
            self.pd2 = nn.Linear(in_features=20*4*4, out_features=10)

            self.multiply1 = Multiply()
            self.multiply2 = Multiply()
            self.softmax = nn.Softmax(dim=1)

            if self.comm_type == "shallow":
                self.communication_layers = [self.fe1,self.pd1]
            elif self.comm_type == "deep":
                self.communication_layers = [self.fe1,self.fe2,self.pd2]
            elif self.comm_type == "mix":
                self.communication_layers = [self.fe1,self.pd1,self.fe2,self.pd2]

    def forward(self,x):
        if self.model_type=="shallow":
            x = self.fe1(x)
            x = self.flatten1(x)
            x = self.pd1(x)
            x = self.softmax(x)
            return x
        elif self.model_type=="deep":
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.softmax(x, dim=1)
            # x = self.fe1(x)
            # x = self.fe2(x)
            # x = self.flatten2(x)
            # x = self.pd2(x)
            # x = self.softmax(x)
            # return x
        elif self.model_type=="mix":
            x = self.fe1(x)
            mid = self.flatten1(x)
            mid = self.pd1(mid)
            x = self.fe2(x)
            x = self.flatten2(x)
            out = self.pd2(x)
            mid = self.multiply1(mid)
            out = self.multiply2(out)
            out += mid
            out = self.softmax(out)
            return out
        
def flatten(model):
    embedding = []
    for child in model.communication_layers:
        for param in child.parameters():
            embedding.append(param.data.view(-1))
    return torch.cat(embedding)

def unflatten(model,embedding):
    pointer = 0
    embedding = embedding
    for child in model.communication_layers:
        for param in child.parameters():
            num_value = torch.prod(torch.LongTensor(list(param.size())))
            param.data = embedding[pointer:pointer+num_value].view(param.size())
            pointer+=num_value