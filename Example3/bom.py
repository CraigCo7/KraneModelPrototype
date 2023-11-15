import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import gensim.downloader as api
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from schedule import createScheduleDataframe
import random

# bomDf = pd.read_csv("Example 3 BOM - Rack 1 and 4.csv")


def createRackFeature(value):
    return value.split('-')[0].strip()


def createMaterialFeature(value):
    return value.split('-')[1].strip()


def createSystemFeature(value):
    return value.split('-')[2].strip()


def createTestingFeature(value):
    return value.split('-')[3].strip()


def setOutput(value, list1, list2):
    if value == 'R01':
        output = list1.pop()
        list1.insert(0, output)
        return output
    else:
        output = list2.pop()
        list2.insert(0, output)
        return output


def featureEngineer(bom_input_path):
    bomDf = pd.read_csv(bom_input_path)
    bomDf['ID'] = bomDf.reset_index().index + 1

    bomDf['Rack'] = bomDf['Spool'].apply(createRackFeature)
    bomDf['Material'] = bomDf['Spool'].apply(createMaterialFeature)
    bomDf['System'] = bomDf['Spool'].apply(createSystemFeature)

    # Create Outputs - assign rack 1&4 to activities randomly.
    scheduleDf = createScheduleDataframe('Example3ScheduleAdjusted.csv')
    rack1 = scheduleDf.loc[scheduleDf['Rack'] == 'R01']
    rack4 = scheduleDf.loc[scheduleDf['Rack'] == 'R04']

    rack1_activities = rack1['ID'].values.tolist()
    rack4_activities = rack4['ID'].values.tolist()

    bomDf['Output Rack'] = bomDf['Rack'].apply(
        setOutput, args=(rack1_activities, rack4_activities, ))

    # More Feature Engineering

    # bomDf['Testing'] = bomDf['Spool'].apply(createTestingFeature)
    bomDf = pd.get_dummies(bomDf, columns=['Rack'], prefix='Rack')
    bomDf = pd.get_dummies(bomDf, columns=['Material'], prefix='Material')
    bomDf = pd.get_dummies(bomDf, columns=['System'], prefix='System')
    bomDf = pd.get_dummies(bomDf, columns=['Assembly'], prefix='Assembly')
    bomDf = pd.get_dummies(bomDf, columns=['Conn'], prefix='Conn')
    # bomDf = pd.get_dummies(bomDf, columns=['Subsystem'], prefix='Subsystem')
    bomDf = pd.get_dummies(bomDf, columns=['Test Pack'], prefix='Test Pack')

    bomDf.loc[bomDf['Pressure'].str.endswith("PSIG"), 'Pressure'] = bomDf.loc[bomDf['Pressure'].str.endswith(
        "PSIG"), 'Pressure'].str.rstrip('PSIG').astype(float)

    return bomDf

# print(bomDf.columns)
# print(len(bomDf.columns.values))
# print(df.columns)


def kmeansProcessOutput(bom_input_path, num_clusters):
    bomDf = featureEngineer(bom_input_path)
    kmeans = KMeans(n_clusters=num_clusters)
    features = bomDf.iloc[:, 13:]

    X = features

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)

    kmeans.fit(X)
    # kmeans.fit(data[0])
    cluster_assignments = kmeans.labels_
    print(cluster_assignments)
    print(len(cluster_assignments))

    # Create a dictionary to store data points for each cluster
    clustered_data = {cluster_num: [] for cluster_num in range(num_clusters)}

    # Assign data points to their respective clusters
    for i in range(len(X)):
        cluster_num = cluster_assignments[i]
        row = bomDf[bomDf['ID'] == i]
        clustered_data[cluster_num].append(row['Spool'].values)

    # print("Lengths of arrays:")

    # for key, value in clustered_data.items():
    #     print(f"{key}: {len(value)}")

    max_length = max(len(value) for value in clustered_data.values())

    # Fill in with None for arrays with shorter length
    clustered_data_filled = {
        key: value + [None] * (max_length - len(value)) for key, value in clustered_data.items()}

    # print("Lengths of arrays:")

    # for key, value in clustered_data_filled.items():
    #     print(f"{key}: {len(value)}")

    df = pd.DataFrame(clustered_data_filled)

    # Save the DataFrame to a CSV file
    df.to_csv('Example3SpoolClusters.csv', index=False)
    # Iterate through the dictionary and print keys and values
    # for cluster_num, data_points in clustered_data.items():
    #     print(f"Cluster {cluster_num} contains the following data points:")
    #     for data_point in data_points:
    #         # print(data_point)
    #         result = bomDf[bomDf['ID'] == data_point+1]
    #         print(result['Spool'].values)

    #     print()  # Add an empty line to separate clusters


# featureEngineer('Example 3 BOM - Rack 1 and 4.csv')
# kmeansProcessOutput('Example 3 BOM - Rack 1 and 4.csv', 25)
# print(data[0])
# kmeans.fit(data[0])
# kmeans.cluster_centers_
# kmeans.labels_
