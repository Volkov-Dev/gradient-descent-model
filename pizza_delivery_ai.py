# 1: import the needed libraries
import numpy as np 
import os 
import pandas as pd 

# 2: Assigning File paths
dataset_file = "pizza_delivery_data.csv"
model_file = "pizza_delivery_model.npz"

# 3: Load the dataset from a file
if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"Dataset file '{dataset_file}' not found! Please provide the file.")
data = pd.read_csv(dataset_file) # Load the dataset

# 4: Handling missing values
if data.isnull().values.any():
    raise ValueError("Dataset contains missing values. Please clean the dataset.")

# 5: Extracting Data from Columns
distance = data["distance"].values
time_of_order = data["time_of_order"].values
delivery_time = data["delivery_time"].values

# 6: Checking Data Validity
if len(distance) == 0 or len(time_of_order) == 0 or len(delivery_time) == 0:
    raise ValueError("Dataset is empty or improperly formatted.")

# 7: Loading or Initializing Model Parameters
if os.path.exists(model_file):
    model_data = np.load(model_file)
    m_distance = model_data["m_distance"]
    m_time = model_data["m_time"]
    b = model_data["b"]
    print(f"Saved model successfully loaded")
else:
    m_distance, m_time, b = 0, 0, 0
    print("No saved model found. Initializing new parameters.")

# 8: Setting Training Parameters
learning_rate = 0.001
epochs = 1000

# 9: Train the model (only if no saved model is found)
if not os.path.exists(model_file):
    for epoch in range(epochs):
        predicted_time = m_distance * distance + m_time * time_of_order + b
        error = delivery_time - predicted_time

        # Debugging: Check for NaN or unexpected values
        if np.any(np.isnan(error)):
            raise ValueError("ERROR: calcualtion contains NaN values.\n-->> If your dataset size is large, then you should lower the algorithm's 'learning_rate' to avoid anomalies in the calculations.")

# 10: Calculating Gradients and Updating Parameters
        m_distance_gradient = -(2 / len(distance)) * sum(distance * error)
        m_time_gradient = -(2 / len(time_of_order)) * sum(time_of_order * error)
        b_gradient = -(2 / len(delivery_time)) * sum(error)

        m_distance -= learning_rate * m_distance_gradient
        m_time -= learning_rate * m_time_gradient
        b -= learning_rate * b_gradient

# 11: Saving the Trained Model
    np.savez(model_file, m_distance=m_distance, m_time=m_time, b=b)
    print("Model trained and saved to", model_file)

# 12: Making Predictions
new_distance = float(input("Enter the distance to your house in km: "))
new_time_of_order = float(input("Enter the time of order (hour in 24-hour format): "))
predicted_time = m_distance * new_distance + m_time * new_time_of_order + b
print(f"Based on your data, it will take approximately {predicted_time:.2f} minutes for your pizza to arrive!\n")
