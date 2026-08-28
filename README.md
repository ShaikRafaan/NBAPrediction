# 🏀 NBA Player Stats Predictor — AI Scouting & Analytics Suite

A broadcast-grade, modern, and interactive web application powered by **Machine Learning (XGBoost)** to forecast NBA player performance metrics (Points, Assists, Rebounds, Field Goals Made/Attempted, FG%, and Minutes Per Game) based on physical attributes, role, playing time, position, and team context.

---

## ✨ Features & Architecture

- ⚡ **AI Prediction Engine**: Multi-output `XGBRegressor` machine learning model trained on active NBA player datasets.
- 🎨 **Next-Gen Broadcast UI**: Responsive glassmorphism interface with NBA franchise branding, dynamic glows, and micro-interactions.
- 📐 **Dual Unit System**: Instant toggle between **Imperial** (`6'6"`, `215 lbs`) and **Metric** (`198 cm`, `97.5 kg`).
- ⚡ **Archetype Presets**: 1-click loading of player prototypes (*Elite Scoring Guard*, *Playmaking Maestro*, *Dominant Two-Way Forward*, *Triple-Double Big*, *3&D Wing*, *Rim Anchor*).
- 🕸️ **Playstyle Radar Matrix**: Interactive SVG 5-axis spider chart (Scoring, Playmaking, Rebounding, Efficiency, Workhorse index).
- 👥 **Real NBA Player Comps**: Multi-dimensional vector similarity matching against active NBA stars.
- 📊 **Interactive League Scatter Plot**: Real-time canvas scatter visualizer comparing the projected player against active NBA peers with hover tooltips and position filters.
- 📋 **Scouting Benchmark Table**: Side-by-side comparison against position average and league-wide benchmarks.

---

## 🚀 Quickstart Guide

### 1. Activate the Virtual Environment
```bash
# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Run the Full Application (Flask + Built React SPA)
```bash
python app.py
```
Open **http://localhost:5000** in your browser!

---

## 🛠️ Frontend Development Mode (Hot Reloading)

If you want to edit the React frontend with instant Hot Module Replacement (HMR):

1. Start the Flask backend API:
```bash
python app.py
```

2. In a second terminal, start the Vite dev server:
```bash
cd frontend
npm run dev
```
Open **http://localhost:5173**! All `/api` requests will proxy to the Flask backend.

3. To build the production bundle:
```bash
cd frontend
npm run build
```
*(The production build is automatically served by `app.py` at `http://localhost:5000`)*

---

## 📁 Repository Structure

```
NBAPrediction/
├── app.py                      # Flask REST API & Static Server
├── models/
│   └── nba_player_stats_predictor.pkl  # Trained XGBoost & Scaler Model
├── data/
│   └── nba_players_full_stats.csv      # NBA Player Dataset
├── frontend/                   # Modern React + Vite Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx              # Status Bar & Header
│   │   │   ├── InputPanel.jsx          # Sliders, Unit Switcher & Presets
│   │   │   ├── StatHero.jsx            # Key Stats & Percentiles
│   │   │   ├── RadarChart.jsx          # Archetype Spider Matrix
│   │   │   ├── PlayerComparisons.jsx   # Top 4 NBA Comps
│   │   │   ├── ScatterChart.jsx        # Interactive Canvas Scatter Plot
│   │   │   └── LeagueContextTable.jsx  # League & Position Benchmarks
│   │   ├── utils/
│   │   │   └── nbaData.js              # Team Colors, Abbrs & Unit Helpers
│   │   ├── App.jsx                     # Root Dashboard State
│   │   └── index.css                   # Custom NBA Analytics Design System
│   ├── package.json
│   └── vite.config.js
└── venv/                       # Python Virtual Environment
```

---

## 📄 License
MIT License © 2025 [Rafaan Shaik](https://github.com/ShaikRafaan)
