import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path

folder_path = '/content/drive/My Drive/TennisProject/data/'

def read_data_for_embedding_network(folder_path):
  player_names = {}
  player_id2index = {}
  players_n = 0
  embedding_data = []
  pathlist = Path(folder_path).glob('**/*.csv')
  for path in pathlist:
     path_in_str = str(path)
     print(path_in_str)
     raw = pd.read_csv(path_in_str, low_memory=False)
     for i, match in raw.iterrows():
      if match['winner_id'] not in player_names:
        player_names[match['winner_id']] = match['winner_name']
        player_id2index[match['winner_id']] = players_n
        players_n += 1
      if match['loser_id'] not in player_names:
        player_names[match['loser_id']] = match['loser_name']
        player_id2index[match['loser_id']] = players_n
        players_n += 1
      embedding_data.append([player_id2index[match['winner_id']], 
                             player_id2index[match['loser_id']], 
                            int(match['w_ace'] > match['l_ace'])])
  return player_names, player_id2index, players_n, embedding_data

player_names, player_id2index, players_n, embedding_data = read_data_for_embedding_network(folder_path)


"""**Embedding network**"""
class EmbeddingsNet(torch.nn.Module):

  def __init__(self, players_n, hidden_size):
    '''
      players_n - the size of the one hot vectors representing a single player
      hidden_size - the output size of Linear of winner, Linear of loser and Linear 1
    '''
    # define he network architecture 
    super(EmbeddingsNet, self).__init__()
    # initial parameters
    self.players_n=players_n
    self.hidden_size=hidden_size
    # first level
    self.linear_winner = nn.Linear(players_n,hidden_size)
    self.linear_loser = nn.Linear(players_n,hidden_size)
    # linear 1
    self.linear_1 = nn.Linear(2*hidden_size, hidden_size)
    # linear 2
    self.linear_2 = nn.Linear(hidden_size, 2) 

  def forward(self, player_1_batch, player_2_batch):
    '''
      player_1_batch - Bx(number of players) tensor - a batch of winner one hot vectors
      player_2_batch - Bx(number of players) tensor - a batch of loser one hot vectors
    '''
    # create 2 input linear layers
    player_1_batch=self.linear_winner(player_1_batch)
    player_2_batch=self.linear_loser(player_2_batch)
    # concat both linear layers into 1 dimension vector
    concat_batch=torch.cat((player_1_batch, player_2_batch),1)
    # use Relu activation function on linear 1
    concat_batch=F.relu(self.linear_1(concat_batch))
    concat_batch=self.linear_2(concat_batch)
    return concat_batch

"""**Training the embedding network**"""

# Intializing the embedding network:
embedding_net = EmbeddingsNet(players_n, 200).cuda()
print(embedding_net)


def train_embeddings_network(model, data, epochs, batch_size):
  # define the parameters
  trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True)
  optimizer = torch.optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss()
  loss_per_epoch = []

  # train network function
  for epoch in range(epochs):
    running_loss = 0.0
    running_loss_per_epoch= 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        winner, loser, labels = data

        # For GPU
        winner=winner.cuda()
        loser=loser.cuda()
        labels=labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # create one hot vector
        winner=F.one_hot(winner, model.players_n).float()
        loser=F.one_hot(loser, model.players_n).float()

        # forward
        outputs = model(winner, loser)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_loss_per_epoch+=loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0 
 
    loss_per_epoch.append(running_loss_per_epoch / 6000)

  # print the loss curve
  print('Loss Curve')
  epochArr=np.arange(0,40)
  plt.plot(epochArr, np.asarray(loss_per_epoch), c='blue')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

# call the training procedure you implemented:
train_embeddings_network(embedding_net, embedding_data, 40, 8)


params=embedding_net.parameters()
matricies=[param for param in params]
# weight matrix of the first two linear layers:
matrix_1=matricies[0]
matrix_2=matricies[2]
player_embedding=(matrix_1+matrix_2)/2
print("Tensor shape:" ,player_embedding.shape)  # 2053 players, each player represented by 200 feaures
#print(player_embedding[:,3]) #embedding vector representing player of index 3
#prints the matrix that represent all the players embedding vectors
print(player_embedding)


"""**Prediction network**"""

class PredictionNetwork(torch.nn.Module):
  def __init__(self, players_embedding_size, match_feature_size, hidden_size1, hidden_size2, hidden_size3):
    # define he network architecture 
    super(PredictionNetwork, self).__init__()
	
    # initial parameters
    self.players_embedding_size = players_embedding_size
    self.match_feature_size =match_feature_size
    self.hidden_size1= hidden_size1
    self.hidden_size2 = hidden_size2
    self.hidden_size3 = hidden_size3
    
	# net architecture:
    # first level
    self.linear_player_1 = nn.Linear(players_embedding_size,hidden_size1)
    self.linear_player_2 = nn.Linear(players_embedding_size,hidden_size1)
    self.linear_match_features = nn.Linear(match_feature_size,hidden_size1)
    # linear 1
    self.linear_1 = nn.Linear(3*hidden_size1,hidden_size2)
    # linear 2
    self.linear_2 = nn.Linear(hidden_size2,hidden_size3)
    # linear 3 
    self.linear_3 = nn.Linear(hidden_size3,2)

  def forward(self, player1_batch, player2_batch, match_features_batch):
    '''
      player1_batch - Bx(players_embedding_size) tensor - a batch of player 1 embedding vectors
      player2_batch - Bx(players_embedding_size) tensor - a batch of player 2 embedding vectors
      match_features_batch - Bx(match_feature_size) tensor - a batch of match-level features
    '''
    # create 3 linear input layers 
    player1_batch = F.relu(self.linear_player_1(player1_batch))
    player2_batch = F.relu(self.linear_player_2(player2_batch))
    match_features_batch = F.relu(self.linear_match_features(match_features_batch))
    # concat the 3 linear layers into 1 dimension vector
    concat_batch=torch.cat((player1_batch,player2_batch,match_features_batch),1)
    # fisrt linear layer
    concat_batch=F.relu(self.linear_1(concat_batch))
    # second linear layer
    concat_batch=F.relu(self.linear_2(concat_batch))
    # third linear layer
    concat_batch=self.linear_3(concat_batch)
    return concat_batch


#prepares the dataset of instances required to train our prediction network
def prepare_match_dataset(folder_path, player_embedding):
  X = []
  y = []
  surface_dictionary = {}
  pathlist = Path(folder_path).glob('**/*.csv')
  for path in pathlist:
    path_in_str = str(path)
    print(path_in_str)
    raw = pd.read_csv(path_in_str, low_memory=False)
    for i, match in raw.iterrows():
      instance = []
      features = []
      features.append(match['draw_size'])
      winner_embedding = player_embedding[:, player_id2index[match['winner_id']]].cpu().detach().numpy()
      loser_embedding = player_embedding[:, player_id2index[match['loser_id']]].cpu().detach().numpy()
      target = -1
      if random.random() < 0.5:
         instance.append(winner_embedding)
         instance.append(loser_embedding)
         target = 0
      else:
         instance.append(loser_embedding)
         instance.append(winner_embedding)
         target = 1
      instance.append(features)
      X.append(instance)
      y.append(target)

  return X, y

X, y = prepare_match_dataset(folder_path, player_embedding)

"""**Training the prediction network**"""

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

def train_match_network(model, X_train, X_test, y_train, y_test, epochs, batch_size):
  # define loss function and optimizers
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  # define the arrays
  train_loss_per_epoch = []
  test_accuracy_per_epoch = []

  # define the trainloader and testloader 
  train_data=[[x,y] for x,y in zip(X_train,y_train)]
  trainloader=torch.utils.data.DataLoader(train_data, batch_size=batch_size)
  test_data=[[x,y] for x,y in zip(X_test,y_test)]
  testloader=torch.utils.data.DataLoader(test_data, batch_size=batch_size)

  for epoch in range(epochs):
    running_loss_per_epoch = 0.0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # for GPU
        players_1=inputs[0].cuda()
        players_2=inputs[1].cuda()
        match=inputs[2][0].cuda()
        labels = labels.cuda()

        # create 1 vector array in 'match_feature_size' size and 'match data'
        match=match.reshape(-1,model.match_feature_size).float()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model(players_1,players_2,match)
        loss = criterion(outputs, labels)
        # backword
        loss.backward()
        # optimize
        optimizer.step()

        # print and plot the loss function for the train data 
        running_loss += loss.item()
        running_loss_per_epoch+=loss.item()
        if (i + 1) % 3000 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 3000))
            running_loss = 0.0

    train_loss_per_epoch.append(running_loss_per_epoch/18000)
        
    # test set accuracy:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
          # get the inputs
          inputs, labels = data
          # for GPU
          players_1=inputs[0].cuda()
          players_2=inputs[1].cuda()
          match=inputs[2][0].cuda()
          labels = labels.cuda()

          # create 1 vector array in 'match_feature_size' size and match data
          match=match.reshape(-1,model.match_feature_size).float()
          
          # calcualte acuuracy
          outputs = model(players_1,players_2,match)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    # print accuracy per epoch
    print('Epoch %d Accuracy: %d %%' % (epoch+1, (100 * correct / total)))
    test_accuracy_per_epoch.append(100 * correct / total)
  
  # print the train loss curve
  print('Train Loss Curve')
  epochArr=np.arange(0,40)
  plt.plot(epochArr, np.asarray(train_loss_per_epoch), c='blue')
  #plt.legend(['Loss', 'Accuracy'], loc='upper right')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

  # print the test accuracy curve
  print('Test Accuracy Curve')
  epochArr=np.arange(0,40)
  plt.plot(epochArr, np.asarray(test_accuracy_per_epoch), c='red')
  #plt.legend(['Loss', 'Accuracy'], loc='upper right')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.show()

tennisMatchModel = PredictionNetwork(200, 1, 1024, 700, 200).cuda()
train_match_network(tennisMatchModel, X_train, X_test, y_train, y_test, 40, 2)