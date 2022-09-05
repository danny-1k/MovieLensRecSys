import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import UserMovieRatings

from models import UserMovieModel


trainloader = DataLoader(UserMovieRatings(), batch_size=256, shuffle=True)
testloader = DataLoader(UserMovieRatings(False), batch_size=256, shuffle=True)

no_users = 138493
no_movies = 131262

user_embed_dim = 64
movie_embed_dim = 64

# One beauty of Deep Matrix Factorization is that the embeddings of 
# say the users and movies don't need to be of the same dimensions because we're not working
# with dot products anymore (not literally)

net = UserMovieModel(no_users=no_users,
    no_movies=no_movies,
    user_embed_dim=user_embed_dim,
    movie_embed_dim=movie_embed_dim)


lossfn = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=1e-3)


net.fit(
    epochs=50,
    lossfn=lossfn,
    optimizer=optimizer,
    trainloader=trainloader,
    testloader=testloader,
    plot_dir='usermoviemodel/loss_plot.png',
    model_dir='usermoviemodel/model.pt'
)