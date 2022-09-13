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

    def get_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'


    def handle_trainloader(self, trainloader, train_loss_epoch, optimizer, lossfn):
        for x, y in trainloader:

            loss = self.one_pass_trainloader(x, y, lossfn)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            train_loss_epoch.append(loss.item())

    
    def handle_testloader(self, testloader, test_loss_epoch, lossfn):
        with torch.no_grad():
            for x, y in testloader:

                loss = self.one_pass_testloader(x, y, lossfn)

                test_loss_epoch.append(loss.item())


    def one_pass_trainloader(self, x, y, lossfn):

        x = [x_i.to(self.device) for x_i in x] if type(x) == list else x.to(self.device)
        y = y.to(self.device)

        p = self.__call__(*x)


        loss = lossfn(p, y)

        return loss


    def one_pass_testloader(self, x, y, lossfn):

        x = [x_i.to(self.device) for x_i in x] if type(x) == list else x.to(self.device)
        y = y.to(self.device)

        p = self.__call__(*x)


        loss = lossfn(p, y)

        return loss



    def fit(self, epochs,lossfn, optimizer, trainloader, testloader, plot_dir=None, model_dir=None, verbosity=0):

        if not verbosity == -1:
            print('Started Training')

        device = self.get_device()
        self.device = device

        self.to(device)


        if not verbosity == -1:

            if device == 'cuda':
                print('Using GPU')
            else:
                if verbosity == 1:
                    print('Using CPU')




        self.train_loss_over_time = []
        self.test_loss_over_time = []

        lowest_loss = float('inf')
        
        for epoch in range(epochs):

            train_loss_epoch = []
            test_loss_epoch = []


            self.handle_trainloader(trainloader, train_loss_epoch, optimizer, lossfn)

            self.handle_testloader(testloader, test_loss_epoch, lossfn)

                
            train_loss = sum(train_loss_epoch)/len(train_loss_epoch)
            test_loss = sum(test_loss_epoch)/len(test_loss_epoch)

            self.train_loss_over_time.append(train_loss)
            self.test_loss_over_time.append(test_loss)


            if plot_dir:


                plt.title('Train & Test Loss Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')

                plt.plot(self.train_loss_over_time, label='Train Loss')
                plt.plot(self.test_loss_over_time, label='Test Loss')

                plt.legend()

                plt.savefig(plot_dir)
                plt.close('all')


            if test_loss < lowest_loss:

                if not verbosity == -1:
                
                    print(f'EPOCH : {epoch+1} TRAIN-LOSS : {train_loss :.3f} TEST-LOSS : {test_loss :.3f}')

                lowest_loss = test_loss

                if model_dir:

                    torch.save(self.state_dict(), model_dir)

                if not verbosity == -1:

                    print('------------CHECKPOINT----------')
            
            if verbosity == 1:
                print(f'EPOCH : {epoch+1} TRAIN-LOSS : {train_loss :.3f} TEST-LOSS : {test_loss :.3f}')




class UserMovieModel(nn.Module, Model):
    def __init__(self, no_users, no_movies, user_embed_dim, movie_embed_dim, hidden_dim=100):
        super().__init__()

        self.no_users = no_users
        self.no_movies = no_movies

        self.user_embed_dim = user_embed_dim
        self.movie_embed_dim = movie_embed_dim

        self.user_embed = nn.Embedding(no_users, user_embed_dim)
        self.movie_embed = nn.Embedding(no_movies, movie_embed_dim)

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(user_embed_dim+movie_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x1, x2):
        x1 = self.user_embed(x1)
        x2 = self.movie_embed(x2)

        x = torch.cat([x1, x2], axis=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)


        return x



class UserMovieCategeoryModel(nn.Module, Model):
    def __init__(self, no_users, no_movies, no_categories, user_embed_dim, movie_embed_dim, category_embed_dim, hidden_dim=100):
        super().__init__()

        self.no_users = no_users
        self.no_movies = no_movies
        self.no_categories = no_categories

        self.user_embed_dim = user_embed_dim
        self.movie_embed_dim = movie_embed_dim
        self.category_embed_dim = category_embed_dim

        self.user_embed = nn.Embedding(no_users, user_embed_dim)
        self.movie_embed = nn.Embedding(no_movies, movie_embed_dim)
        self.category_embed = nn.Embedding(no_categories, category_embed_dim)

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(user_embed_dim+movie_embed_dim+category_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x1, x2, x3):
        x1 = self.user_embed(x1)
        x2 = self.movie_embed(x2)
        x3 = self.category_embed(x3)

        x = torch.cat([x1, x2, x3], axis=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)


        return x


class UserMovieModelImplicit(nn.Module, Model):
    def __init__(self, no_users, no_movies, user_embed_dim, movie_embed_dim, hidden_dim=100):
        super().__init__()

        self.no_users = no_users
        self.no_movies = no_movies

        self.user_embed_dim = user_embed_dim
        self.movie_embed_dim = movie_embed_dim

        self.user_embed = nn.Embedding(no_users, user_embed_dim)
        self.movie_embed = nn.Embedding(no_movies, movie_embed_dim)

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(user_embed_dim+movie_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x1, x2):
        x1 = self.user_embed(x1)
        x2 = self.movie_embed(x2)

        x = torch.cat([x1, x2], axis=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)


        return x


    def negative_sample_pred(self, user_ids):

        """Randomly sample the items with hope that a significant percent of
        the sampled items have not been interacted with by the users represented
        by `user_ids`.
        """

        negative_items = torch.randint(0, self.no_movies, user_ids.shape)

        return self.__call__(user_ids, negative_items)

