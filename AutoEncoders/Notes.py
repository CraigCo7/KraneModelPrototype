# Stacked AutoEncoder (SAE)

"""
Preprocessing pulled from RBM lecture
"""
# Import Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import Data
movies = pd.read_csv('ml-1m/movies.dat',
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat',
                    sep = '::',
                    header = None,
                    engine = 'python',
                    encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/movies.dat',
                      sep = '::',
                      header = None,
                      engine = 'python',
                      encoding = 'latin-1')

# Prepare Training and Test Sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int64')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int64')

# Quantify Users and Movies
qty_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
qty_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Convert To Array
def convert(data):
    new_data = []
    for id_users in range(1, qty_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(qty_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Torch Data
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
"""
End preprocessing from RBM lecture
"""

# Build SAE
"""
NOTE: opportunity to make variable for number of neurons on each layer
"""
class SAE(nn.Module):   # Stacked AutoEncoder inherits parent class parameters
    def __init__(self, ):
        super(SAE, self).__init__()   # inherits all classes and modules
        self.fc1 = nn.Linear(qty_movies, 20)   # fc1 set to first full connection related to AE, 20 neurons arbitrary and needs improvement
        self.fc2 = nn.Linear(20, 10)   # 2nd hidden layer set to 10 neurons, again arbitrary
        self.fc3 = nn.Linear(10, 20)   # 20 neurons for a symmetrical network
        self.fc4 = nn.Linear(20, qty_movies)   # input/output neurons qty = qty_movies
        self.activation = nn.Sigmoid()   # sigmoid activation function
    def forward(self, x):   # forward pass will be encoded twice (input>20>10) and decoded twice (10>20>output)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(),
                          lr = 0.01,   # find better learning rate
                          weight_decay = 0.5)   # find better weight

# Train SAE
qty_epoch = 200
for epoch in range(1, qty_epoch + 1):
    train_loss = 0
    s_count = 0.   # use float to prevent warnings
    for id_users in range(qty_users):
        input = Variable(training_set[id_users]).unsqueeze(0)   # keras cannot accept 1D vector as input, need to create batch
        target = input.clone()   # duplicates the input, will change target later
        if torch.sum(target.data > 0) > 0:   # if function to exclude users who did not rate movies
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = qty_movies/float(torch.sum(target.data > 0) + 1e-10)   # prevent divide by zero
            loss.backward()   # direction of weight update
            train_loss += np.sqrt(loss.data*mean_corrector)   # take train_loss from loss data
            s_count += 1
            optimizer.step()   # intensity of weight update
    print('epoch: '+str(epoch)+' - loss: '+str(train_loss/s_count))

# Test SAE
test_loss = 0
s_count = 0.
for id_users in range(qty_users):
    input = Variable(training_set[id_users]).unsqueeze(0)   # training set has all user's rated movies, intend to predict movies not yet rated
    target = Variable(test_set[id_users]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = qty_movies/float(torch.sum(target.data > 0) + 1e-10)
        #loss.backward()   # not needed as this is for backpropagation
        test_loss += np.sqrt(loss.data*mean_corrector)
        s_count += 1
        #optimizer.step()   # backpropagation
print('test loss: '+str(test_loss/s_count))

# Results
"""
Output: test loss 0.9468
This means that the model predicts what the user will rate
a movie with accuracy of +- 1 star
"""















