import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import gensim.downloader as api
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from feature import addFeaturesForProcessOutput
# Importing the dataset
# df = pd.read_csv('Guard Shack - Schedule 4WLA.csv')
df = pd.read_csv('CE Input/GuardShackSchedule.csv')
model = api.load("word2vec-google-news-300")
# model = api.load("conceptnet-numberbatch-17-06-300")

hardcodedProcesses = ["Base", "Wall", "Roof", "Electrical"]


def kmeansProcessOutput():
    addFeaturesForProcessOutput(df, hardcodedProcesses)
    kmeans = KMeans(n_clusters=4)
    features = ['cosineSimilarityComparison(Base)', 'cosineSimilarityComparison(Wall)',
                'cosineSimilarityComparison(Roof)', 'cosineSimilarityComparison(Electrical)']
    # features = ['cosineSimilarity(Base)', 'cosineSimilarity(Wall)',
    #             'cosineSimilarity(Roof)', 'cosineSimilarity(Electrical)']
    X = df[features]

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)

    kmeans.fit(X)
    # kmeans.fit(data[0])
    cluster_assignments = kmeans.labels_
    print(cluster_assignments)

    # Create a dictionary to store data points for each cluster
    clustered_data = {cluster_num: [] for cluster_num in range(4)}

    # Assign data points to their respective clusters
    for i in range(len(X)):
        cluster_num = cluster_assignments[i]
        clustered_data[cluster_num].append(i)

    # Iterate through the dictionary and print keys and values
    for cluster_num, data_points in clustered_data.items():
        print(f"Cluster {cluster_num} contains the following data points:")
        for data_point in data_points:
            # print(data_point)
            result = df[df['ID'] == data_point+1]
            print(result['Activity Name'].values)

        print()  # Add an empty line to separate clusters


kmeansProcessOutput()
# print(data[0])
# kmeans.fit(data[0])
# kmeans.cluster_centers_
# kmeans.labels_
