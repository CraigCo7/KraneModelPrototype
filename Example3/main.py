from matplotlib import pyplot as plt
from schedule import createScheduleDataframe
from bom import featureEngineer
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from sklearn.cluster import KMeans
import numpy as np

csv_input = 'Example3ScheduleAdjusted.csv'


def k_means_autoencoder():
    # Import BOM Data
    bomDf = featureEngineer('Example 3 BOM - Rack 1 and 4.csv')

    print(bomDf.columns.values)
    print(bomDf.shape)

    # Define the autoencoder architecture
    # input_layer = Input(shape=(2,))
    # encoded = Dense(1, activation='relu')(input_layer)
    # decoded = Dense(2, activation='sigmoid')(encoded)

    # autoencoder = Model(input_layer, decoded)
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')


k_means_autoencoder()
