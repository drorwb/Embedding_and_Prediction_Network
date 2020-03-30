# Embedding and Prediction Network

In this project i've implemnted 2 networks, Embedding and Prediction networks, in order to predict the winner of a given Tennis match.

### Dataset

The [dataset](https://drive.google.com/open?id=1ie6WSl8qkknpSGAFt32Gj7Tf0JAhTLu0) is composed of a collection of Tennis matches and their outcomes, provided in multiple files. 

### Embedding network

The embedding network trained to predict a single outcome, given two tennis players, in order to learn how to represent a player using a vector.

![Embedding network](https://imgur.com/RcVtemW.png)

### Prediction network

Given two tennis players and some additional match-leavel features, this network predicts who is about to win. 

Each player is represented by a vector that is learned by the embedding network.

![Prediction network](https://i.imgur.com/9cHIVlE.png)


# Try it yourself

You can check out the running result of my project in the following [link](https://colab.research.google.com/drive/15Tld8huhfADCi9MzQywrXgm7d5OlVKd4).

You can make a copy of this notebook and try it yourself. Just don't forget to change the "folder_path" to where you saved the dataset in your drive.
