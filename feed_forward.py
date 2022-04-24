from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#! device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#! hyper parameters:
input_size = 784 #28*28
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.001

#! mnist
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
train_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

example = iter(train_loader)
samples, labels = example.next()
print(samples.shape)
print(labels.shape)


for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap = 'gray')

# plt.show()

class NeuralNet(nn.module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

#! Loss:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters = model.parameters(), lr = learning_rate)

#! train_loops:
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #! forward pass:
        outputs = model(images)
        loss = criterion(outputs, labels)

        #! backward pass:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'current epoch : {epoch+1}/{num_epochs}, step : {i+1}/{n_total_steps}, loss : {loss.item():.4f}')

