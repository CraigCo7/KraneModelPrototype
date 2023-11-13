# SOM to help divide the activities into Major Processes in the Schedule 4WLA.

# Importing the libraries
from pylab import bone, pcolor, colorbar, plot, show
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import cosine_similarity
import torch
from word2vec import word2vec, sentence2vec
import gensim.downloader as api
import torch.nn.functional as F
from feature import addFeaturesForProcessOutput, addFeaturesForProcessStepSuccessor


# Importing the dataset
# df = pd.read_csv('Guard Shack - Schedule 4WLA.csv')
df = pd.read_csv('CE Input/GuardShackSchedule.csv')
model = api.load("word2vec-google-news-300")
# model = api.load("conceptnet-numberbatch-17-06-300")

hardcodedProcesses = ["Base", "Walls", "Roof", "Electrical"]


# Convert the numpy arrays to PyTorch tensors
# word_vector = torch.tensor(
#     word_vector).clone().detach().requires_grad_(True)
# sentence_vector = torch.tensor(
#     sentence_vector).clone().detach().requires_grad_(True)


def somProcessOutput():
    addFeaturesForProcessOutput(df, hardcodedProcesses)

    # Choose features
    features = ['cosineSimilarity(Base)', 'cosineSimilarity(Walls)',
                'cosineSimilarity(Roof)', 'cosineSimilarity(Electrical)']
    X = df[features]  # features

    y = df['ID']  # labels

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)

    # Training the SOM

    # Low learning rate (<=0.1) = more stable output
    # Low sigma (<=0.5) = more focused and localized neighborhood (high sigma means larger neighborhood)
    learningRate = 0.05
    sigma = 0.5
    som = MiniSom(x=2, y=2, input_len=len(features),
                  sigma=sigma, learning_rate=learningRate)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)

    # Initialize a dictionary to store cluster assignments
    cluster_assignments = {}

    # Iterate through your data and assign each data point to its corresponding cluster
    for i in range(len(X)):
        winner = som.winner(X[i])
        if winner not in cluster_assignments:
            cluster_assignments[winner] = [i]
        else:
            cluster_assignments[winner].append(i)

    # Now, you can access the cluster assignments as needed
    print("LEARNING: ", learningRate)
    print("SIGMA: ", sigma)
    for cluster, data_points in cluster_assignments.items():
        print(f"Cluster {cluster}: {data_points}", )


# # Get the final cluster assignments for each data point (VISUALIZE)
# cluster_assignments = np.zeros(
#     (10, 10), dtype=int)  # Assuming a 10x10 SOM grid

# for i in range(len(X)):
#     winner = som.winner(X[i])
#     cluster_assignments[winner] += 1

# # Create a heatmap to display the final cluster assignments
# plt.figure(figsize=(10, 10))
# plt.pcolor(cluster_assignments, cmap='coolwarm')
# plt.colorbar()

# # Show the plot
# plt.title('Final Cluster Assignments')
# plt.show()

somProcessOutput()
