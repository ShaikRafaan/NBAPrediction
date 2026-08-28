import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

# Determine base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'nba_player_stats_predictor.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'nba_players_full_stats.csv')
DIST_DIR = os.path.join(BASE_DIR, 'frontend', 'dist')

# Load trained model, scaler, and feature columns
model, scaler, feature_cols, numeric_cols = joblib.load(MODEL_PATH)
player_data = pd.read_csv(CSV_PATH)

# Clean dataset for comparison and ranking
player_data_clean = player_data[
    (player_data['gp'] > 0) & 
    (player_data['min'] > 0)
].copy()

player_data_clean['PTS_pg'] = np.round(player_data_clean['pts'] / player_data_clean['gp'], 2)
player_data_clean['AST_pg'] = np.round(player_data_clean['ast'] / player_data_clean['gp'], 2)
player_data_clean['REB_pg'] = np.round(player_data_clean['reb'] / player_data_clean['gp'], 2)
player_data_clean['FGM_pg'] = np.round(player_data_clean['fgm'] / player_data_clean['gp'], 2)
player_data_clean['FGA_pg'] = np.round(player_data_clean['fga'] / player_data_clean['gp'], 2)
player_data_clean['FG%'] = np.round(player_data_clean['fg_pct'], 3)
player_data_clean['MIN_pg'] = np.round(player_data_clean['min'] / player_data_clean['gp'], 1)

# Extract unique dropdown values
unique_teams = sorted([t for t in player_data['team'].dropna().unique() if str(t).strip() != ''])
unique_positions = sorted([p for p in player_data['position'].dropna().unique() if str(p).strip() != ''])

# NBA Archetype Presets for rapid exploration
ARCHETYPES = [
    {
        "id": "scoring-guard",
        "title": "Elite Scoring Guard",
        "description": "High volume, crafty shot creator (e.g. Shai Gilgeous-Alexander, Devin Booker)",
        "tag": "Scorer",
        "height_cm": 198,
        "weight_kg": 90,
        "gp": 75,
        "min": 2550,
        "position": "Guard",
        "team": "Thunder" if "Thunder" in unique_teams else unique_teams[0]
    },
    {
        "id": "point-god",
        "title": "Playmaking Maestro",
        "description": "Floor general with supreme vision and passing (e.g. Tyrese Haliburton, Trae Young)",
        "tag": "Playmaker",
        "height_cm": 193,
        "weight_kg": 86,
        "gp": 70,
        "min": 2400,
        "position": "Guard",
        "team": "Pacers" if "Pacers" in unique_teams else unique_teams[0]
    },
    {
        "id": "freak-forward",
        "title": "Dominant Two-Way Forward",
        "description": "Unstoppable physical rim runner and paint attacker (e.g. Giannis Antetokounmpo)",
        "tag": "All-NBA",
        "height_cm": 211,
        "weight_kg": 110,
        "gp": 72,
        "min": 2450,
        "position": "Forward",
        "team": "Bucks" if "Bucks" in unique_teams else unique_teams[0]
    },
    {
        "id": "playmaking-big",
        "title": "Triple-Double Big",
        "description": "Offensive engine hub with high efficiency and elite rebounding (e.g. Nikola Jokić)",
        "tag": "MVP",
        "height_cm": 211,
        "weight_kg": 128,
        "gp": 78,
        "min": 2700,
        "position": "Center",
        "team": "Nuggets" if "Nuggets" in unique_teams else unique_teams[0]
    },
    {
        "id": "three-and-d",
        "title": "3 & D Wing Specialist",
        "description": "High motor, efficient floor spacer and versatile defender (e.g. OG Anunoby, Derrick White)",
        "tag": "Sharpshooter",
        "height_cm": 201,
        "weight_kg": 98,
        "gp": 76,
        "min": 2350,
        "position": "Forward-Guard" if "Forward-Guard" in unique_positions else "Forward",
        "team": "Celtics" if "Celtics" in unique_teams else unique_teams[0]
    },
    {
        "id": "paint-anchor",
        "title": "Rim Protecting Anchor",
        "description": "Glass cleaner, lob threat, and defensive anchor (e.g. Jarrett Allen, Rudy Gobert)",
        "tag": "Anchor",
        "height_cm": 216,
        "weight_kg": 118,
        "gp": 74,
        "min": 2300,
        "position": "Center",
        "team": "Cavaliers" if "Cavaliers" in unique_teams else unique_teams[0]
    }
]

app = Flask(__name__, static_folder=DIST_DIR, template_folder='templates')

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/')
def home():
    if os.path.exists(os.path.join(DIST_DIR, 'index.html')):
        return send_from_directory(DIST_DIR, 'index.html')
    return render_template('index.html', team=unique_teams, position=unique_positions)

@app.route('/<path:path>')
def static_proxy(path):
    if os.path.exists(os.path.join(DIST_DIR, path)):
        return send_from_directory(DIST_DIR, path)
    if os.path.exists(os.path.join(DIST_DIR, 'index.html')):
        return send_from_directory(DIST_DIR, 'index.html')
    return send_from_directory('static', path)

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Returns teams, positions, archetypes, and dataset statistics."""
    return jsonify({
        "status": "success",
        "teams": unique_teams,
        "positions": unique_positions,
        "archetypes": ARCHETYPES,
        "bounds": {
            "height_cm": {"min": 170, "max": 226, "default": 200},
            "weight_kg": {"min": 65, "max": 145, "default": 95},
            "gp": {"min": 1, "max": 82, "default": 70},
            "min": {"min": 50, "max": 3100, "default": 2200}
        },
        "total_players_tracked": len(player_data_clean)
    })

def calculate_percentile(value, series):
    """Calculates the percentile rank (0-100) of a value against a pandas Series."""
    valid = series.dropna()
    if len(valid) == 0:
        return 50.0
    rank = (valid < value).mean() * 100.0
    return float(np.round(np.clip(rank, 1.0, 99.9), 1))

def find_closest_nba_players(predicted_per_game, position, top_k=4):
    """Finds real NBA players in the dataset with the closest statistical profile."""
    # Filter qualified players (at least 15 games and 250 minutes)
    pool = player_data_clean[
        (player_data_clean['gp'] >= 15) & 
        (player_data_clean['min'] >= 250)
    ].copy()

    if len(pool) == 0:
        pool = player_data_clean.copy()

    # Target vector: [PTS_pg, AST_pg, REB_pg, FG%]
    pts_pred = predicted_per_game.get('PTS', 10.0)
    ast_pred = predicted_per_game.get('AST', 2.5)
    reb_pred = predicted_per_game.get('REB', 4.0)
    fg_pct_pred = predicted_per_game.get('FG%', 0.45)

    # Weights for distance calculation
    # Normalized by realistic standard deviations
    std_pts = max(float(pool['PTS_pg'].std()), 1.0)
    std_ast = max(float(pool['AST_pg'].std()), 1.0)
    std_reb = max(float(pool['REB_pg'].std()), 1.0)
    std_fg = max(float(pool['FG%'].std()), 0.05)

    dists = np.sqrt(
        1.5 * ((pool['PTS_pg'] - pts_pred) / std_pts) ** 2 +
        1.2 * ((pool['AST_pg'] - ast_pred) / std_ast) ** 2 +
        1.2 * ((pool['REB_pg'] - reb_pred) / std_reb) ** 2 +
        1.0 * ((pool['FG%'] - fg_pct_pred) / std_fg) ** 2
    )

    # Boost same position slightly
    same_pos_mask = pool['position'].str.contains(position.split('-')[0], case=False, na=False)
    dists[same_pos_mask] *= 0.85

    pool['dist'] = dists
    top_matches = pool.sort_values(by='dist').head(top_k)

    comps = []
    for _, row in top_matches.iterrows():
        # Similarity score between 60% and 99%
        similarity = float(np.round(np.clip(100.0 / (1.0 + row['dist'] * 0.35), 65.0, 99.4), 1))
        comps.append({
            "name": str(row.get('player_x') or row.get('player_y') or row.get('player') or "NBA Player"),
            "team": str(row.get('team', 'NBA')),
            "position": str(row.get('position', 'Forward')),
            "similarity": similarity,
            "pts_pg": float(row['PTS_pg']),
            "ast_pg": float(row['AST_pg']),
            "reb_pg": float(row['REB_pg']),
            "fg_pct": float(row['FG%']),
            "min_pg": float(row['MIN_pg']),
            "gp": int(row['gp'])
        })
    return comps

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if request.is_json:
            data = request.get_json()
            height_cm = float(data.get('height_cm', 200))
            weight_kg = float(data.get('weight_kg', 95))
            gp = float(data.get('gp', 70))
            min_played = float(data.get('min', 2200))
            team = str(data.get('team', unique_teams[0] if unique_teams else 'Lakers'))
            position = str(data.get('position', unique_positions[0] if unique_positions else 'Guard'))
        else:
            height_cm = float(request.form.get('height_cm', 200))
            weight_kg = float(request.form.get('weight_kg', 95))
            gp = float(request.form.get('gp', 70))
            min_played = float(request.form.get('min', 2200))
            team = str(request.form.get('team', unique_teams[0] if unique_teams else 'Lakers'))
            position = str(request.form.get('position', unique_positions[0] if unique_positions else 'Guard'))

        gp = max(1.0, gp)
        min_played = max(1.0, min_played)

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

        input_df = pd.DataFrame([input_data])

        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Scale numeric features
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        prediction = model.predict(input_df)
        prediction = np.maximum(prediction, 0)
        prediction = np.round(prediction[0], 2)

        # Total season prediction
        stat_names = ['PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG%']
        totals = dict(zip(stat_names, [float(x) for x in prediction]))

        # Per-game stats
        per_game_stats = {}
        for key, value in totals.items():
            if key != 'FG%':
                per_game_stats[key] = round(float(value / gp), 2)
            else:
                # Keep FG% realistic (clamped between 0.30 and 0.72)
                per_game_stats[key] = round(float(np.clip(value, 0.300, 0.720)), 3)

        min_pg = round(float(min_played / gp), 1)
        per_game_stats['MIN'] = min_pg

        # Calculate Percentiles (League & Position)
        pos_df = player_data_clean[player_data_clean['position'] == position]
        if len(pos_df) < 5:
            pos_df = player_data_clean

        percentiles_league = {
            'PTS': calculate_percentile(per_game_stats['PTS'], player_data_clean['PTS_pg']),
            'AST': calculate_percentile(per_game_stats['AST'], player_data_clean['AST_pg']),
            'REB': calculate_percentile(per_game_stats['REB'], player_data_clean['REB_pg']),
            'FG%': calculate_percentile(per_game_stats['FG%'], player_data_clean['FG%']),
            'MIN': calculate_percentile(min_pg, player_data_clean['MIN_pg'])
        }

        percentiles_position = {
            'PTS': calculate_percentile(per_game_stats['PTS'], pos_df['PTS_pg']),
            'AST': calculate_percentile(per_game_stats['AST'], pos_df['AST_pg']),
            'REB': calculate_percentile(per_game_stats['REB'], pos_df['REB_pg']),
            'FG%': calculate_percentile(per_game_stats['FG%'], pos_df['FG%']),
            'MIN': calculate_percentile(min_pg, pos_df['MIN_pg'])
        }

        # Player Archetype / Radar Ratings (0-100 scale)
        radar_ratings = [
            {"category": "Scoring", "value": round(float(np.clip(percentiles_league['PTS'] * 0.85 + (per_game_stats['PTS'] / 32.0) * 15, 10, 99)), 1), "fullMark": 100},
            {"category": "Playmaking", "value": round(float(np.clip(percentiles_league['AST'] * 0.85 + (per_game_stats['AST'] / 11.0) * 15, 10, 99)), 1), "fullMark": 100},
            {"category": "Rebounding", "value": round(float(np.clip(percentiles_league['REB'] * 0.85 + (per_game_stats['REB'] / 14.0) * 15, 10, 99)), 1), "fullMark": 100},
            {"category": "Efficiency", "value": round(float(np.clip(percentiles_league['FG%'], 10, 99)), 1), "fullMark": 100},
            {"category": "Workhorse", "value": round(float(np.clip((min_pg / 38.0) * 100, 10, 99)), 1), "fullMark": 100}
        ]

        # Closest NBA Player Comparisons
        player_comps = find_closest_nba_players(per_game_stats, position, top_k=4)

        # Overall composite rating / Tier
        avg_score = (per_game_stats['PTS'] * 1.5 + per_game_stats['AST'] * 2.0 + per_game_stats['REB'] * 1.5 + per_game_stats['FG%'] * 40.0) / 1.6
        tier = "All-NBA Superstar" if avg_score > 65 else "NBA All-Star" if avg_score > 48 else "High-Impact Starter" if avg_score > 32 else "Key Rotation Player" if avg_score > 20 else "Role Player / Prospect"

        return jsonify({
            "status": "success",
            "inputs": {
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "gp": int(gp),
                "min": int(min_played),
                "min_pg": min_pg,
                "team": team,
                "position": position
            },
            "per_game": per_game_stats,
            "totals": totals,
            "percentiles": {
                "league": percentiles_league,
                "position": percentiles_position
            },
            "radar_ratings": radar_ratings,
            "comparisons": player_comps,
            "composite": {
                "score": round(float(np.clip(avg_score, 15.0, 99.0)), 1),
                "tier": tier
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/scatter-data', methods=['GET'])
def get_scatter_data():
    """Returns dataset points for interactive scatter charts by position."""
    position = request.args.get('position', None)
    
    df = player_data_clean.copy()
    if position and position != 'All':
        df = df[df['position'] == position]
    
    # Filter reasonable minutes to avoid zero outliers
    df = df[df['min'] > 50]
    
    data_points = []
    for _, row in df.iterrows():
        data_points.append({
            "name": str(row.get('player_x') or row.get('player_y') or row.get('player') or "NBA Player"),
            "team": str(row.get('team', '')),
            "position": str(row.get('position', '')),
            "min": float(row['min']),
            "min_pg": float(row['MIN_pg']),
            "pts_pg": float(row['PTS_pg']),
            "ast_pg": float(row['AST_pg']),
            "reb_pg": float(row['REB_pg']),
            "fgm_pg": float(row['FGM_pg']),
            "fga_pg": float(row['FGA_pg']),
            "fg_pct": float(row['FG%']),
            "gp": int(row['gp'])
        })
        
    return jsonify({
        "status": "success",
        "count": len(data_points),
        "position": position or "All",
        "data": data_points
    })

# Legacy predict route for backwards compatibility
@app.route('/predict', methods=['POST'])
def legacy_predict():
    return api_predict()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
