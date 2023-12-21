import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Load data
melbourne_file_path = './curso_kaggle_intro_to_ml/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

# Define model
melbourne_model = DecisionTreeRegressor(random_state=1)

# Split data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Fit model
melbourne_model.fit(train_X, train_y)

val_predictions = melbourne_model.predict(val_X)

print(f"\n\nPrevisão:\n{val_predictions}")
print(f"\n\nval_y:\n{val_y}")

mae = mean_absolute_error(val_y, val_predictions)

print(f"\n\nmae:\n{mae}")