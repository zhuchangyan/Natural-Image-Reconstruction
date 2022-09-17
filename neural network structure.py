

# SHL-DNN,   used in natural_ML.py

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(112*112, Nnodes)
        self.fc2 = nn.Linear(Nnodes, 64*64)
        self.dp = nn.Dropout(0.2)
        self.BN = nn.BatchNorm1d(Nnodes)
        self.BN2 = nn.BatchNorm1d(64*64)  
        self.BN3 = nn.BatchNorm2d(10)  
        self.ct1 = nn.ConvTranspose2d(1,10,2,stride = 2)
        self.ct2 = nn.ConvTranspose2d(10,6,2,stride = 2)
        
        self.conv2 = nn.Conv2d(6,1,3)
        self.pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(3969, 64*64)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.BN(self.dp(F.leaky_relu(self.fc1(x))))
        x = self.BN2(self.dp(F.sigmoid(self.fc2(x))))
        x = x.view(-1,1,64,64)
        # x = F.relu(self.ct1(x))
        # x = F.relu(self.ct2(x))
        # x = self.pool(self.conv2(x))
        # x = self.flatten(x)
        # x = self.dp(F.leaky_relu(self.fc3(x)))
        # x = x.view(-1,1,64,64)
        return x


# Single layer neural network to extract real Tranmission matrix, used in Temperature_test.ipynb

import math

class complexLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights1 = torch.Tensor(size_out, size_in)
        self.weights1 = nn.Parameter(weights1)  # nn.Parameter is a Tensor that's a module parameter.
#         weights2 = torch.Tensor(size_out, size_in)
#         self.weights2 = nn.Parameter(weights2)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights1, a=math.sqrt(5)) # weight init
#         nn.init.kaiming_uniform_(self.weights2, a=math.sqrt(5)) # weight init


    def forward(self, x):
        x1 = torch.mm(x,self.weights1.t())
#         x1 = torch.square(torch.mm(torch.cos(x),self.weights1.t()))
#         x2 = torch.square(torch.mm(torch.sin(x),self.weights2.t()))
#         x =  torch.add(x1,x2)
        return x1

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.my1 = complexLayer(112*112, out_dim*out_dim)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.my1(x)
        x = x.view(-1, 1, out_dim, out_dim)
        return x

model = NeuralNetwork().to(device)    






#  U-net, used in SHL_U_net.ipynb
#  U-net structure is not optimized here, it's also not used in my production of temperature test results.


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(112*112, Nnodes)
        self.fc2 = nn.Linear(Nnodes, 64*64)
        self.dp = nn.Dropout(0.2)
        self.BN = nn.BatchNorm1d(Nnodes)
        self.BN2 = nn.BatchNorm1d(64*64)  
        self.BN3 = nn.BatchNorm2d(10)  
        self.ct1 = nn.ConvTranspose2d(1,10,2,stride = 2)
        self.ct2 = nn.ConvTranspose2d(10,6,2,stride = 2)
        
        self.conv2 = nn.Conv2d(6,1,3)
        self.pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(3969, 64*64)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.BN(self.dp(F.leaky_relu(self.fc1(x))))
        x = self.BN2(self.dp(F.sigmoid(self.fc2(x))))
        x = x.view(-1,1,64,64)
        # x = F.relu(self.ct1(x))
        # x = F.relu(self.ct2(x))
        # x = self.pool(self.conv2(x))
        # x = self.flatten(x)
        # x = self.dp(F.leaky_relu(self.fc3(x)))
        # x = x.view(-1,1,64,64)
        return x