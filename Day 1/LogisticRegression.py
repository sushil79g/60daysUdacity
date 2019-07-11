import torch
import torch.nn as nn
from torch.autograd import Variable

x = torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0]])
y = torch.Tensor([[1],[0],[1],[0],[1],[0]])

class LogisticRegression:
    def __init__(self, input_size, no_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, no_classes)

    def forward(self, x):
        output = self.linear(x)
        return output

model = LogisticRegression(1,1)
criterion = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epochs in range(50):
    prediction = model(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('for epoch {} loss={}'.format(epochs+1, loss.item()))

test = torch.Tensor([[7.0]])
pred = model(test)
print("model prediction for {} is {}".format(7, pred))