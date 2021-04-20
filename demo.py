import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ftrl import FTRL

# Hyperparameters
batch_size = 100

input_size = 64
output_size = 10

ftrl_alpha = 1.0
ftrl_beta = 1.0
ftrl_l1 = 1.0
ftrl_l2 = 1.0


# Dataset
from sklearn.datasets import load_digits


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.W = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.W(x)


model = LogisticRegression(input_size, output_size)
loss_fn = nn.CrossEntropyLoss()

optimizer = FTRL(
    model.parameters(), alpha=ftrl_alpha, beta=ftrl_beta, l1=ftrl_l1, l2=ftrl_l2
)

# Train
X,y=load_digits(10,True)
optimizer.zero_grad()
y_pred=model(torch.from_numpy(X.astype('float32')))
y_tensor=torch.from_numpy(y.astype('int64'))
loss=loss_fn(y_pred,y_tensor)
loss.backward()
optimizer.step()
