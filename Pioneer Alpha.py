import torch
from torch import nn
import torch.nn.functional as F

model = nn.Sequential(nn.Linear(2, 4),
                      nn.ReLU(),
                      nn.Linear(4, 1),
                      nn.Sigmoid())

criterion = nn.BCELoss()

# Random x and y
x = torch.randn(10, 2)
y = (torch.sigmoid(torch.randn(10, 1)) >= .5)*1.0

y_hat = model(x)

loss = criterion(y_hat, y)
loss.backward()
