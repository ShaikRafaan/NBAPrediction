import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import numpy as np

# Load dataset
data = pd.read_csv("data/nba_players_full_stats.csv")

# Convert height to centimeters
def height_to_cm(height_str):
    if pd.isnull(height_str):
        return None
    feet, inches = height_str.split('-')
    return (int(feet) * 12 + int(inches)) * 2.54

data['height_cm'] = data['height'].apply(height_to_cm)

# Convert weight to kilograms
data['weight_kg'] = data['weight'] * 0.453592

# Drop unnecessary columns
columns_to_drop = ['player_id', 'player_x', 'player_y', 'season', 'team_id', 'height', 'weight']
data.drop(columns=columns_to_drop, inplace=True)

# One-hot encode team and position
data = pd.get_dummies(data, columns=['team', 'position'], drop_first=False)

# Define features and targets
feature_cols = ['height_cm', 'weight_kg', 'gp', 'min'] + \
               [col for col in data.columns if col.startswith('team_') or col.startswith('position_')]
target_cols = ['pts', 'ast', 'reb', 'fgm', 'fga', 'fg_pct']

X = data[feature_cols]
y = data[target_cols]

# Scale numeric features only
numeric_cols = ['height_cm', 'weight_kg', 'gp', 'min']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
base_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=4)
model = MultiOutputRegressor(base_model)

# Fit model
model.fit(X_train, y_train)

# Save model, scaler, and feature columns
joblib.dump((model, scaler, feature_cols, numeric_cols), 'models/nba_player_stats_predictor.pkl')


print("✅ Model, scaler, and feature columns saved successfully.")
