import torch
import torch.nn as nn
from torch.autograd import Variable

x = torch.Tensor([ [1.0], [2.0], [3.0], [4.0]])
y = torch.Tensor([ [1.0], [8.0], [27.0], [64.0]])

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epochs in range(50):
    prediction = model(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('for epoch {} loss={}'.format(epochs+1, loss.item()))

test = torch.Tensor([[5.0]])
pred = model(test)
print("model prediction for {} is {}".format(5, pred))