from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load trained model, scaler, feature columns
model, scaler, feature_cols, numeric_cols = joblib.load('models/nba_player_stats_predictor.pkl')
csv_path = 'data/nba_players_full_stats.csv'
player_data = pd.read_csv(csv_path)
# Unique teams and positions for dropdowns
unique_teams = sorted(player_data['team'].dropna().unique())
unique_positions = sorted(player_data['position'].dropna().unique())

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', team=unique_teams, position=unique_positions)

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
    prediction = np.maximum(prediction, 0)
    prediction = np.round(prediction[0], 2)

    # Convert prediction to per-game stats
    results = dict(zip(['PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG%'], prediction))
    per_game_stats = {}
    for key, value in results.items():
        if key != 'FG%':
            per_game_stats[key] = round(value / gp, 2)
        else:
            per_game_stats[key] = round(value, 2)

    # Prepare comparison data
    player_data_clean = player_data[
        (player_data['gp'] > 0) &
        (player_data['min'] > 0) &
        (player_data['position'] == position)
    ].copy()

    player_data_clean['PTS_pg'] = player_data_clean['pts'] / player_data_clean['gp']
    player_data_clean['AST_pg'] = player_data_clean['ast'] / player_data_clean['gp']
    player_data_clean['REB_pg'] = player_data_clean['reb'] / player_data_clean['gp']
    player_data_clean['FGM_pg'] = player_data_clean['fgm'] / player_data_clean['gp']
    player_data_clean['FGA_pg'] = player_data_clean['fga'] / player_data_clean['gp']
    player_data_clean['FG%'] = player_data_clean['fg_pct']

    # Create plots
    stats_to_plot = ['PTS_pg', 'AST_pg', 'REB_pg', 'FGM_pg', 'FGA_pg', 'FG%']
    plot_paths = []

    for stat in stats_to_plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(player_data_clean['min'], player_data_clean[stat], alpha=0.5, label='Other Players')

        # Add predicted point
        stat_key = stat if stat == 'FG%' else stat.split('_')[0]
        predicted_value = per_game_stats[stat_key]
        plt.scatter(min_played, predicted_value, color='red', label='Predicted Player', s=100, edgecolors='black')

        plt.xlabel('Minutes Played')
        plt.ylabel(stat.replace('_pg', ' Per Game') if stat != 'FG%' else 'Field Goal %')
        plt.title(f'{stat.replace("_pg", "").upper()} vs Minutes Played ({position})')
        plt.legend()
        plt.grid(True)

        filename = f'static/{stat.lower().replace("%", "pct")}_vs_min.png'
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        plot_paths.append(filename)

    return render_template('results.html', results=per_game_stats, scatter_paths=plot_paths)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
