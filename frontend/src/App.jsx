import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import Navbar from './components/Navbar';
import InputPanel from './components/InputPanel';
import StatHero from './components/StatHero';
import RadarChart from './components/RadarChart';
import PlayerComparisons from './components/PlayerComparisons';
import ScatterChart from './components/ScatterChart';
import LeagueContextTable from './components/LeagueContextTable';
import { Sparkles, AlertCircle, RefreshCw } from 'lucide-react';

const API_BASE = '/api';

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [scatterData, setScatterData] = useState([]);
  const [unitSystem, setUnitSystem] = useState('imperial'); // 'imperial' | 'metric'
  const [activeArchetypeId, setActiveArchetypeId] = useState('scoring-guard');
  
  const [formState, setFormState] = useState({
    height_cm: 198,
    weight_kg: 90,
    gp: 75,
    min: 2550,
    team: 'Thunder',
    position: 'Guard'
  });

  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Fetch initial metadata and scatter points
  useEffect(() => {
    async function init() {
      try {
        const metaRes = await fetch(`${API_BASE}/metadata`);
        if (!metaRes.ok) throw new Error('Failed to connect to backend API');
        const metaData = await metaRes.json();
        setMetadata(metaData);
        setIsOnline(true);

        if (metaData.archetypes && metaData.archetypes.length > 0) {
          const defaultArch = metaData.archetypes[0];
          setActiveArchetypeId(defaultArch.id);
          setFormState({
            height_cm: defaultArch.height_cm,
            weight_kg: defaultArch.weight_kg,
            gp: defaultArch.gp,
            min: defaultArch.min,
            team: defaultArch.team,
            position: defaultArch.position
          });
          // Run initial prediction with default preset
          runPrediction({
            height_cm: defaultArch.height_cm,
            weight_kg: defaultArch.weight_kg,
            gp: defaultArch.gp,
            min: defaultArch.min,
            team: defaultArch.team,
            position: defaultArch.position
          });
        }

        // Fetch scatter points
        const scatterRes = await fetch(`${API_BASE}/scatter-data`);
        if (scatterRes.ok) {
          const scatterJson = await scatterRes.json();
          setScatterData(scatterJson.data || []);
        }
      } catch (err) {
        console.error("Initialization error:", err);
        setErrorMsg("Could not connect to the NBA Predictor API. Ensure the backend server is running.");
        setIsOnline(false);
      }
    }
    init();
  }, []);

  // Run prediction API call
  const runPrediction = async (customState = null) => {
    const payload = customState || formState;
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.message || 'Prediction failed');
      }

      const data = await res.json();
      setPrediction(data);
      setIsOnline(true);

      // Trigger confetti if high tier superstar
      if (data.composite && data.composite.score >= 65) {
        confetti({
          particleCount: 50,
          spread: 60,
          origin: { y: 0.7 },
          colors: ['#f97316', '#38bdf8', '#fbbf24', '#ffffff']
        });
      }
    } catch (err) {
      console.error("Prediction error:", err);
      setErrorMsg(err.message || "Failed to generate prediction. Please check inputs.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navbar */}
      <Navbar
        isOnline={isOnline}
        totalPlayers={metadata?.total_players_tracked || scatterData.length}
      />

      {/* Main Container */}
      <main style={{
        maxWidth: '1400px',
        width: '100%',
        margin: '0 auto',
        padding: '24px',
        flex: 1
      }}>
        
        {/* Error Alert */}
        {errorMsg && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            borderRadius: '12px',
            padding: '14px 18px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            color: '#fca5a5'
          }}>
            <AlertCircle size={20} color="#f87171" />
            <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>{errorMsg}</span>
          </div>
        )}

        {/* Dashboard Main Grid Layout */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(340px, 420px) 1fr',
          gap: '24px',
          alignItems: 'start'
        }}>
          
          {/* Left Column: Input Form & Attribute Controls */}
          <div style={{ position: 'sticky', top: '90px' }}>
            <InputPanel
              formState={formState}
              setFormState={setFormState}
              unitSystem={unitSystem}
              setUnitSystem={setUnitSystem}
              archetypes={metadata?.archetypes || []}
              teams={metadata?.teams || []}
              positions={metadata?.positions || []}
              onPredict={() => runPrediction()}
              isLoading={isLoading}
              activeArchetypeId={activeArchetypeId}
              setActiveArchetypeId={setActiveArchetypeId}
            />
          </div>

          {/* Right Column: Projections, Visualizations & Insights */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            
            {/* Stat Projection Hero Card */}
            <StatHero
              prediction={prediction}
              formState={formState}
            />

            {/* Radar Playstyle & Real Player Comparisons Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
              gap: '24px'
            }}>
              <RadarChart radarData={prediction?.radar_ratings} />
              <PlayerComparisons
                comparisons={prediction?.comparisons}
                projectedStats={prediction?.per_game}
              />
            </div>

            {/* Interactive Scatter Plot */}
            <ScatterChart
              predictedStat={prediction}
              formState={formState}
              scatterData={scatterData}
              position={formState.position}
            />

            {/* Benchmark vs League Table */}
            <LeagueContextTable
              prediction={prediction}
              position={formState.position}
              scatterData={scatterData}
            />

          </div>
        </div>

      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-color)',
        padding: '24px',
        textAlign: 'center',
        color: 'var(--text-dim)',
        fontSize: '0.8rem',
        marginTop: '40px',
        background: 'rgba(9, 13, 22, 0.95)'
      }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <span>🏀 NBA Player Stats Predictor • Powered by XGBoost & Machine Learning</span>
          <span>Trained on Active NBA Player Dataset</span>
        </div>
      </footer>
    </div>
  );
}
