import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

import seaborn
seaborn.set_style('whitegrid')
import matplotlib.pyplot as plt


torch.manual_seed(42)

no_users = 250
no_items = 250

no_data_points = 35000

data = torch.randn(no_users, 25) @ torch.randn(25, no_items) # generating synthetic user rating data using matrix factorization

row_idxs = np.random.randint(no_users, size=no_data_points)
col_idxs = np.random.randint(no_items, size=no_data_points)

ratings = data[row_idxs, col_idxs]

# print(ratings.shape)

# plt.title('Distribution of ratings')
# plt.xlabel('Count')
# plt.ylabel('Value')

# plt.hist(ratings.numpy(), bins=100)
# plt.show()

class MatrixFactorization(nn.Module):
    def __init__(self, no_users, no_items, embed_size):
        super().__init__()

        self.no_users = no_users
        self.no_items = no_items
        self.embed_size = embed_size

        self.user_embed = nn.Embedding(no_users, embed_size)
        self.item_embed = nn.Embedding(no_items, embed_size)

    def forward(self,user_id, item_id):

        x1 = self.user_embed(user_id)
        x2 = self.item_embed(item_id)

        pred = (x1 * x2).sum(axis=1) # dot product between user embedding and item embedding (1, embed_Size) @ (25, embed_size)

        return pred



class MFDataSet(Dataset):
    def __init__(self, train=True):
        self.train = train

        if (train):
            self.user_ids = row_idxs[:25000]
            self.item_ids = col_idxs[:25000]
            self.ratings = ratings[:25000]
            
        else:
            self.user_ids = row_idxs[25000:]
            self.item_ids = col_idxs[25000:]
            self.ratings = ratings[25000:]

    def __getitem__(self, index):
        x1 = self.user_ids[index]
        x2 = self.item_ids[index]
        y = self.ratings[index]

        return (x1, x2, y)


    def __len__(self):
        return self.user_ids.shape[0]



def train(net, epochs, optimizer, lossfn, trainloader, testloader):

    train_loss_over_time = []
    test_loss_over_time = []

    for epoch in range(epochs):

        average_train_loss = []
        average_test_loss = []

        for x1, x2, y in trainloader:

            pred = net(x1, x2)

            loss = lossfn(pred,y)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            
            average_train_loss.append(loss.item())



        with torch.no_grad():

            for x1, x2, y in testloader:
                pred = net(x1, x2)

                loss = lossfn(pred,y)

                average_test_loss.append(loss.item())

        average_train_loss = sum(average_train_loss)/len(average_train_loss)
        average_test_loss = sum(average_test_loss)/len(average_test_loss)

        train_loss_over_time.append(average_train_loss)
        test_loss_over_time.append(average_test_loss)


        print(f'EPOCH : {epoch+1} TRAIN LOSS : {train_loss_over_time[-1]} TEST LOSS : {test_loss_over_time[-1]} (TRAIN-TEST)% : {-((train_loss_over_time[-1]-test_loss_over_time[-1])/test_loss_over_time[-1])*100 :.2f}')



    return train_loss_over_time, test_loss_over_time

trainloader = DataLoader(MFDataSet(train=True), batch_size=64, shuffle=True)
testloader =  DataLoader(MFDataSet(train=True), batch_size=64, shuffle=True)

net = MatrixFactorization(no_users, no_items, 150)
optimizer = optim.Adam(net.parameters(), lr=1e-3,)
lossfn = nn.MSELoss()

train_loss_over_time, test_loss_over_time = train(net, 40, optimizer, lossfn, trainloader, testloader)


plt.title('Loss over time')
plt.xlabel('Epochs')
plt.ylabel('MSE')

plt.plot(train_loss_over_time, label='Train Loss')
plt.plot(test_loss_over_time, label='Test Loss')

plt.legend()


plt.show()

reconstructed = net.user_embed.weight@net.item_embed.weight.T


fig, axes = plt.subplots(2,1)

axes[0].scatter(x=reconstructed.detach().numpy()[0, :], y=reconstructed.detach().numpy()[:, 0])
axes[1].scatter(x=data.detach().numpy()[0, :], y=data.detach().numpy()[:, 0])

plt.show()