# Assuming you have pandas, scikit-learn, and TensorFlow installed
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

# Assuming you have a DataFrame df with columns: 'Activity' and 'OutputActivity' (replace with your actual column names)
# Example DataFrame creation:
df = pd.DataFrame({
    'Activity': ['Form post', 'Form level', 'Form deck', 'Mechanical electrical plumbing hangers', 'Embed anchor bolts', 'Concrete'],
    'OutputActivity': ['Install forms', 'Install forms', 'Install forms', 'Install mechanical electrical plumbing', 'Install rebar and embed items', 'Pour concrete']
})

# Tokenize the text data using bag-of-words representation
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['Activity']).toarray()

print(X)

# Access feature names
feature_names = vectorizer.get_feature_names_out()

print(feature_names)
# Use the output activities as the target variable
y = pd.factorize(df['OutputActivity'])[0]
# y = df['OutputActivity'][0]
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Embedding(input_dim=len(feature_names)+1,
              output_dim=64, input_length=X.shape[1]),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    # Adjust output dimensions based on the number of unique output activities
    Dense(len(df['OutputActivity'].unique()), activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16,
          validation_data=(X_test, y_test))

# Evaluate the model
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)
print(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
