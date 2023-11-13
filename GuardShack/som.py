# Importing the libraries
from feature import addFeaturesForProcessStepSuccessor, general
import gensim.downloader as api
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom


# df = pd.read_csv('CE Input/Input.csv')
model = api.load("word2vec-google-news-300")


def som(schedule_path):
    df = general(schedule_path)

    features = ['Activity Name']
    X = df[features]
    # Set SOM parameters
    som_rows = 2
    som_columns = 2
    input_dimensions = len(word_vectors[0])  # Dimension of word vectors

    # Initialize SOM
    som = MiniSom(som_rows, som_columns, input_dimensions,
                  sigma=1.0, learning_rate=0.5)


# def somProcessStepSuccessor(schedule_path, process_assignments_path):
#     df = addFeaturesForProcessStepSuccessor(
#         schedule_path, process_assignments_path)

#     features = ['Task1StartDay', 'Task1StartMonth', 'Task1StartYear']
#     X = df[features]  # features

#     # y = df['ID']  # labels

#     # Feature Scaling
#     sc = MinMaxScaler(feature_range=(0, 1))
#     X = sc.fit_transform(X)

#     # Training the SOM

#     # Low learning rate (<=0.1) = more stable output
#     # Low sigma (<=0.5) = more focused and localized neighborhood (high sigma means larger neighborhood)
#     learningRate = 0.05
#     sigma = 0.5
#     som = MiniSom(x=1, y=2, input_len=len(features),
#                   sigma=sigma, learning_rate=learningRate)
#     som.random_weights_init(X)
#     som.train_random(data=X, num_iteration=100)

#     # Initialize a dictionary to store cluster assignments
#     cluster_assignments = {}

#     # Iterate through your data and assign each data point to its corresponding cluster
#     for i in range(len(X)):
#         winner = som.winner(X[i])
#         if winner not in cluster_assignments:
#             cluster_assignments[winner] = [i]
#         else:
#             cluster_assignments[winner].append(i)

#     print(cluster_assignments)
#     # Now, you can access the cluster assignments as needed

#     print("LEARNING: ", learningRate)
#     print("SIGMA: ", sigma)
#     for cluster, data_points in cluster_assignments.items():
#         print(f"Cluster {cluster}: {data_points}", )


# somProcessStepSuccessor('CE Input/GuardShackSchedule.csv',
#                         'CE Input/GuardShackProcessOutput.csv')
