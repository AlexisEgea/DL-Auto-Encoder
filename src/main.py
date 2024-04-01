import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from Auto_Encoder import Auto_Encoder
from Train_Data import train
from Test_Data import test
from Dataset import CustomDataset

from Variable.variable import er_8psk, debug
from Display_Constellation import plot_er_ebn0_8psk, plot_constellation_Auto_Encoder_M_PSK, plot_er_ebn0

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

# Uncomment next line if the final most probable value of the estimated message is needed
# final_most_probable_value = test(dataloader=test_dataloader, model=model, device=device)

print("Comparison with M-PSK")

with torch.no_grad():
    x, _, _ = model(tensor.float())
    if debug:
        print(x)

plot_constellation_Auto_Encoder_M_PSK(x, size_M)

ebn0_range = np.arange(0, 15, 1)
# I have only an example of error rates for an 8-PSK (see variable.py file)
if size_M == 8:
    ebnodb_8psk = np.linspace(0, 14, 15)
    plot_er_ebn0_8psk(model, test_dataloader, ebn0_range, er_8psk, ebnodb_8psk, device)
else:
    plot_er_ebn0(model, test_dataloader, ebn0_range, device)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Error Rate")
    plt.title("Error Rates Auto-Encoder")
    plt.grid(True)
    plt.legend()
    plt.show()

# Possibility to save the model and reuse it with (ŝ), but it is not the goal of this project
