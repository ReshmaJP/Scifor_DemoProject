import streamlit as st
import time
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define selected features globally
selected_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
                      "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"]

# Set dark theme for Streamlit
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1E1E1E;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data()
def load_data():
    # Your existing code
    # Your existing code
    # Load Breast Cancer dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Extract only the selected features
    df_selected_features = df[selected_features]

    return df_selected_features, data.target

@st.cache_data()
def build_model(X_train, y_train, X_test, y_test):
    # Your existing code
    # Your existing code

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and compile the model
    N, D = X_train.shape
    model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(D,)), tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

    return model, r, scaler

# Waiting time for the App to display
with st.spinner('Loading...'):
    time.sleep(10)
st.success('')

# Streamlit App
st.title("Breast Cancer Classification using Deep Learning")

# Load data and build model
df_selected_features, target = load_data()
X_train, X_test, y_train, y_test = train_test_split(df_selected_features, target, test_size=0.33)
model, r, scaler = build_model(X_train, y_train, X_test, y_test)

# Display the dataset
st.subheader("Variable Information")
st.dataframe(df_selected_features)

# Display the target values 0,1
st.text('In the Dataset, Target Values are Classified as:')
st.text('Target 0: Benign (Not Cancerous)')
st.text('Target 1: Malignant (Cancerous)')

# Display model training results
st.subheader("Model Training Results")
st.line_chart(pd.DataFrame(r.history))

# Input Section
st.subheader("Prediction Section")

# Example input fields (customize based on your features)
input_features = {}

for feature in selected_features:
    min_val = float(df_selected_features[feature].min())
    max_val = float(df_selected_features[feature].max())
    default_val = float(df_selected_features[feature].mean())
    input_features[feature] = st.slider(f"{feature} Input", min_val, max_val, default_val)

# Make Prediction
prediction_button = st.button("Make Prediction")

if prediction_button:
    # Format the input data for prediction
    input_data = scaler.transform([[input_features[feature] for feature in selected_features]])
    
    # Make prediction
    prediction = model.predict(input_data)[0, 0]
    
    # Display the prediction result
    result_text = "The Tumor is Classified as Malignant (Cancerous)" if prediction > 0.5 else "The Tumor is Classified as Benign (Not Cancerous)"
    st.subheader("Prediction Result:")
    st.write(result_text)

# Plot accuracy and loss over epochs
st.subheader("Accuracy and Loss Over Epochs")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(r.history['accuracy'], label='Train Accuracy')
ax1.plot(r.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend()

ax2.plot(r.history['loss'], label='Train Loss')
ax2.plot(r.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.legend()

# Display the plots in Streamlit
st.pyplot(fig)






