import os
import shutil
import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from models import Model, UserMovieModel, UserMovieCategeoryModel


class ModelTest(unittest.TestCase):

    def setUp(self):

        class TestModel(nn.Module, Model):
            def __init__(self):
                super().__init__()

                self.fc1 = nn.Linear(1,1)

            def forward(self,x):
                x = self.fc1(x)

                return x

        
        self.net = TestModel()

        self.train = DataLoader(TensorDataset(torch.ones((4, 1)), torch.ones((4, 1))))
        self.test = DataLoader(TensorDataset(torch.ones((4, 1)), torch.ones((4, 1))))
        

    def test_get_device(self):
        self.assertTrue(self.net.get_device()=='cpu')


    def test_save_plot_model(self):
        if not os.path.exists('test_plot_dir'):
            os.makedirs('test_plot_dir')

        self.net.fit(1, nn.MSELoss(), optim.SGD(self.net.parameters(), lr=1e-4), self.train, self.test, 'test_plot_dir/loss.png', 'test_plot_dir/model.pt', verbosity=-1)

        self.assertTrue('loss.png' in os.listdir('test_plot_dir') and 'model.pt' in os.listdir('test_plot_dir'))


        shutil.rmtree('test_plot_dir')


    def test_training_loop(self):

        self.net.fit(10, nn.MSELoss(), optim.SGD(self.net.parameters(), lr=1e-3), self.train, self.test, verbosity=-1)

        self.assertLess(self.net.test_loss_over_time[-1],
                                     sum(self.net.test_loss_over_time[:-1])/len(self.net.test_loss_over_time[:-1]))


class UserMovieModelTest(unittest.TestCase):
    def setUp(self):

        self.net = UserMovieModel(3, 3, 3, 3, hidden_dim=3)

        class RecDataset(Dataset):
            def __init__(self):
                
                self.data = [ # userid, movieid, rating
                    [1, 1, 5],
                    [0, 2, 2],
                    [2, 1, 3]
                ]


            def __len__(self):
                return 3

            def __getitem__(self, index):
                item = self.data[index]

                return (item[0], item[1]), torch.Tensor([item[2]/5])

        
        self.train = DataLoader(RecDataset())
        self.test = DataLoader(RecDataset())


    def test_loss_reducing(self):

        self.net.fit(5, nn.MSELoss(), optim.SGD(self.net.parameters(), lr=1e-3), self.train, self.test, verbosity=-1)

        self.assertLess(self.net.test_loss_over_time[-1],
                                     sum(self.net.test_loss_over_time[:-1])/len(self.net.test_loss_over_time[:-1]))


class UserMovieCategoryModelTest(unittest.TestCase):
    def setUp(self):

        self.net = UserMovieCategeoryModel(3, 3, 3, 3, 3, 3, hidden_dim=3)

        class RecDataset(Dataset):
            def __init__(self):
                
                self.data = [ # userid, movieid, rating
                    [1, 1, 1, 5],
                    [0, 2, 0, 2],
                    [2, 1, 2, 3]
                ]


            def __len__(self):
                return 3

            def __getitem__(self, index):
                item = self.data[index]

                return (item[0], item[1], item[2]), torch.Tensor([item[3]/5])


        self.train = DataLoader(RecDataset())
        self.test = DataLoader(RecDataset())


    def test_loss_reducing(self):
        self.net.fit(5, nn.MSELoss(), optim.SGD(self.net.parameters(), lr=1e-3), self.train, self.test, verbosity=-1)

        self.assertLess(self.net.test_loss_over_time[-1],
                                     sum(self.net.test_loss_over_time[:-1])/len(self.net.test_loss_over_time[:-1]))



if __name__ == "__main__":
    unittest.main()