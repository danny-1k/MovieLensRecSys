from turtle import forward
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')



class Model:

    def __init__(self):
        pass

    def forward(self):
        pass

    def __call__(self):
        pass

    def fit(self, epochs,lossfn, optimizer, trainloader, testloader, plot_dir, model_dir):

        print('Started Training')

        
        train_loss_over_time = []
        test_loss_over_time = []

        lowest_loss = float('inf')
        
        for epoch in range(epochs):

            train_loss_epoch = []
            test_loss_epoch = []

            for x, y in trainloader:
                p = self.__call__(*x)


                loss = lossfn(p, y)


                loss.backward()

                optimizer.step()

                optimizer.zero_grad()

                train_loss_epoch.append(loss.item())


            with torch.no_grad():
                for x, y in testloader:
                    p = self.__call__(*x)

                    loss = lossfn(p, y)

                    train_loss_epoch.append(loss.item())
                
                
                test_loss_epoch.append(loss.item())


            train_loss = sum(train_loss_epoch)/len(train_loss_epoch)
            test_loss = sum(test_loss_epoch)/len(test_loss_epoch)

            train_loss_over_time.append(train_loss)
            test_loss_over_time.append(test_loss)


            plt.title('Train & Test Loss Over Time')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plt.plot(train_loss_over_time, label='Train Loss')
            plt.plot(test_loss_over_time, label='Test Loss')

            plt.legend()

            plt.savefig(plot_dir)
            plt.close('all')


            if test_loss < lowest_loss:
                
                print(f'EPOCH : {epoch+1} TRAIN-LOSS : {train_loss :.3f} TEST-LOSS : {test_loss :.3f}')

                lowest_loss = test_loss

                torch.save(self.state_dict(), model_dir)


                print('------------CHECKPOINT----------')



class UserMovieModel(nn.Module, Model):
    def __init__(self, no_users, no_movies, user_embed_dim, movie_embed_dim):
        super().__init__()

        self.no_users = no_users
        self.no_movies = no_movies

        self.user_embed_dim = user_embed_dim
        self.movie_embed_dim = movie_embed_dim

        self.user_embed = nn.Embedding(no_users, user_embed_dim)
        self.movie_embed = nn.Embedding(no_movies, movie_embed_dim)

        self.fc1 = nn.Linear(user_embed_dim+movie_embed_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x1, x2):
        x1 = self.user_embed(x1)
        x2 = self.movie_embed(x2)

        x = torch.cat([x1, x2], axis=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.sigmoid(x)


        return x

