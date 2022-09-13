import pandas as pd

import torch
from torch.utils.data import Dataset


class UserMovieRatings(Dataset):
    def __init__(self, train=True, implicit=False):
        if train:
            self.df = pd.read_csv('MovieLens/ratings_train.csv')

        else:
            self.df = pd.read_csv('MovieLens/ratings_test.csv')

        if implicit:
            self.df.drop(['rating'], axis=1)

        
        self.implicit = implicit

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data_point = self.df.iloc[index]

        # in the data, the ID's start from 0

        x1 = data_point['userId'] - 1 
        x2 = data_point['movieId'] - 1 

        if not self.implicit:

            y = torch.Tensor([data_point['rating']/5]) # Normalize between 0 and 1

            return (x1,x2), y

        else:

            return (x1, x2)


class UserMovieCategoriesRatings(Dataset):
    def __init__(self, train=True, implicit=False):
        if train:
            self.df = pd.read_csv('MovieLens/ratings_train.csv')

        else:
            self.df = pd.read_csv('MovieLens/ratings_test.csv')

        if implicit:
            self.df.drop(['rating'], axis=1)


        self.categories = pd.read_csv('MovieLens/movie_categories.csv')
        self.implicit = implicit


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        data_point = self.df.iloc[index]

        # in the data, the ID's start from 0

        x1 = data_point['userId'] - 1
        x2 = data_point['movieId'] - 1

        x3 = self.categories[self.categories['MovieID'] == x2]
        x3 = x3[[c for c in x3.columns if c != 'MovieID']].values

        
        if not self.implicit:

            y = torch.Tensor([data_point['rating']/5]) # Normalize between 0 and 1

            return (x1,x2,x3), y

        else:

            return (x1,x2,x3)

