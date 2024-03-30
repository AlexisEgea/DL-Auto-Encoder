import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from Auto_Encoder import Auto_Encoder
from Train_Data import train
from Test_Data import test
from Dataset import CustomDataset

from Variable.variable import debug

k = 3
size_M = np.power(2, k)
size_N = 100000
# Each random value is associated with a label position for the one-hot method
N = np.random.randint(size_M, size=size_N)
if debug:
    print(N)

# Application of one-hot method to get in each line, only 0 and one 1
matrix = np.zeros([size_N, size_M])
for i in range(0, size_N):
    message = np.zeros(size_M)
    message[N[i]] = 1
    matrix[i] = message
if debug:
    print(matrix)

# Transform matrix in a tensor to use Pytorch methods
tensor = torch.tensor(matrix)
if debug:
    print(tensor)

# Device Init for the execution
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

model = Auto_Encoder(size_M).to(device)
print(model)

# Selection of loss function + optim function + parameters (can be changed to obtain better results=
loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.001
epochs = 50
batch_size = 1000

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 60% for train, 40% for test
train_size = int(0.6 * len(tensor))
test_size = len(tensor) - train_size
train_tensor, test_tensor = torch.utils.data.random_split(tensor, [train_size, test_size])

train_dataset = CustomDataset(train_tensor)
test_dataset = CustomDataset(test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Learning & Testing Process")

for epoch in range(epochs):
    print("Epochs : " + str(epoch + 1) + "/" + str(epochs))
    train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optim=optim, device=device)
    test(dataloader=test_dataloader, model=model, device=device)

# Possibility to save the model and reuse it, but it is not the goal of this project
