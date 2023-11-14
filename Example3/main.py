from matplotlib import pyplot as plt
from schedule import createScheduleDataframe
from bom import featureEngineer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
import numpy as np

csv_input = 'Example3ScheduleAdjusted.csv'


def k_means_autoencoder():
    # Import BOM Data
    bomDf = featureEngineer('Example 3 BOM - Rack 1 and 4.csv')

    print(bomDf.columns.values)
    print(bomDf.shape)
    columns_to_exclude = [
        'ISO', 'Spool', 'Subsystem', 'Pressure', 'Hydro Group', 'Module', 'Assembly', 'Conn', 'EHT', 'Insul', 'Stage']
    selected_columns = bomDf.drop(columns=columns_to_exclude, axis=1)

    selected_columns_np = selected_columns.values.astype(float)

    # Define the autoencoder architecture
    input_layer = Input(
        shape=((len(bomDf.columns.values.tolist())) - len(columns_to_exclude)),)
    encoded = Dense(1, activation='relu')(input_layer)
    decoded = Dense(2, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(selected_columns_np, selected_columns_np, epochs=50, batch_size=32,
                    shuffle=True, validation_split=0.2)

    # Extract the learned representations from the encoder
    encoder_model = Model(input_layer, encoded)
    encoded_data = encoder_model.predict(selected_columns_np)

    # Apply k-means clustering to the learned representations
    kmeans = KMeans(n_clusters=25)
    kmeans.fit(encoded_data)
    cluster_assignments = kmeans.predict(encoded_data)

    print(cluster_assignments)

    # Define the autoencoder architecture
    # input_layer = Input(shape=(2,))
    # encoded = Dense(1, activation='relu')(input_layer)
    # decoded = Dense(2, activation='sigmoid')(encoded)

    # autoencoder = Model(input_layer, decoded)
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')


k_means_autoencoder()
