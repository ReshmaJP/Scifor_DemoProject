# Import all libraries
import tensorflow as tf
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load Data
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data

# Create a DataFrame with the data
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df

df.info()
df.describe()

# Check for missing values after filling
df.isna().sum()

# Check for outliers
# Create a boxplot for each feature, excluding the target variable
num_features = len(df.columns) - 1  # Exclude the target variable
num_cols = 3
num_rows = (num_features // num_cols) + (num_features % num_cols)

plt.figure(figsize=(20, num_rows * 5))

# Skip the target variable
for i, feature in enumerate(df.columns[:-1], 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=df[feature])
    plt.title(feature, fontsize=12)

plt.tight_layout()
plt.show()

# Data Visualization 
# Heatmap of data
plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(), annot = True)
plt.xticks(rotation = 90)

import plotly.express as px

# Define selected features
selected_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
                      "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"]

# Create a DataFrame with only the selected features
df_selected = df[selected_features + ['target']]

# Set template to dark
px.defaults.template = "plotly_dark"

# Plot scatter matrix with plotly
fig = px.scatter_matrix(df_selected, dimensions=selected_features, color='target')
fig.show()

data.target
data.target_names
data.target.shape
data.feature_names

# Splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.33)

N, D = X_train.shape
N, D

# Using Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building Model
model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(D,)), tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# Iteration
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 100)

# Show Train Score and Test Score
print('Train score:', model.evaluate(X_train, y_train))
print('Test score:', model.evaluate(X_test, y_test))

# Plot loss vs val_loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy vs val_accuracy
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()


#Breast Cancer Classification using Deep Learning App - https://reshmajp-scifor.streamlit.app/











