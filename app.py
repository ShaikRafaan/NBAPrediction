from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Load trained model, scaler, feature columns
model, scaler, feature_cols, numeric_cols = joblib.load('models/nba_player_stats_predictor.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    height_cm = float(request.form['height_cm'])
    weight_kg = float(request.form['weight_kg'])
    gp = float(request.form['gp'])
    min_played = float(request.form['min'])

    team = request.form['team']
    position = request.form['position']

    # Build feature dict
    input_data = {
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'gp': gp,
        'min': min_played
    }

    # Add one-hot encoded team and position columns
    for col in feature_cols:
        if col.startswith('team_'):
            input_data[col] = 1 if col == f'team_{team}' else 0
        elif col.startswith('position_'):
            input_data[col] = 1 if col == f'position_{position}' else 0

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Scale numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)

    # Replace negatives with zero
    prediction = np.maximum(prediction, 0)

    # Round predictions
    prediction = np.round(prediction[0], 2)

    results = dict(zip(['PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG%'], prediction))

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
