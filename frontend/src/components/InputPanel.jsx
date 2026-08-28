import React from 'react';
import { 
  Sliders, 
  Sparkles, 
  Ruler, 
  Weight, 
  Calendar, 
  Clock, 
  Shield, 
  Layers, 
  ArrowRight,
  RefreshCw,
  Zap
} from 'lucide-react';
import { 
  NBA_TEAMS_INFO, 
  cmToFeetInches, 
  feetInchesToCm, 
  kgToLbs, 
  lbsToKg 
} from '../utils/nbaData';

export default function InputPanel({
  formState,
  setFormState,
  unitSystem,
  setUnitSystem,
  archetypes,
  teams,
  positions,
  onPredict,
  isLoading,
  activeArchetypeId,
  setActiveArchetypeId
}) {
  const { height_cm, weight_kg, gp, min, team, position } = formState;

  // Imperial conversions
  const ftIn = cmToFeetInches(height_cm);
  const lbs = kgToLbs(weight_kg);
  const minPerGame = gp > 0 ? (min / gp).toFixed(1) : '0.0';

  const selectedTeamInfo = NBA_TEAMS_INFO[team] || { primary: '#f97316', secondary: '#ffffff', abbr: team.substring(0, 3).toUpperCase() };

  const handleArchetypeClick = (archetype) => {
    setActiveArchetypeId(archetype.id);
    setFormState({
      height_cm: archetype.height_cm,
      weight_kg: archetype.weight_kg,
      gp: archetype.gp,
      min: archetype.min,
      team: archetype.team,
      position: archetype.position
    });
  };

  const handleHeightChange = (valInCm) => {
    setActiveArchetypeId(null);
    setFormState(prev => ({ ...prev, height_cm: Number(valInCm) }));
  };

  const handleWeightChange = (valInKg) => {
    setActiveArchetypeId(null);
    setFormState(prev => ({ ...prev, weight_kg: Number(valInKg) }));
  };

  const handleGpChange = (val) => {
    setActiveArchetypeId(null);
    const newGp = Math.max(1, Math.min(82, Number(val)));
    setFormState(prev => ({ ...prev, gp: newGp }));
  };

  const handleMinChange = (val) => {
    setActiveArchetypeId(null);
    const newMin = Math.max(10, Math.min(3200, Number(val)));
    setFormState(prev => ({ ...prev, min: newMin }));
  };

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      {/* Panel Header & Unit Toggle */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '20px',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Sliders size={20} color="#f97316" />
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0, letterSpacing: '-0.01em' }}>
            Player Attributes & Profile
          </h2>
        </div>

        {/* Unit Selector */}
        <div style={{
          display: 'flex',
          background: '#090d16',
          padding: '3px',
          borderRadius: '10px',
          border: '1px solid var(--border-color)'
        }}>
          <button
            onClick={() => setUnitSystem('imperial')}
            style={{
              background: unitSystem === 'imperial' ? '#f97316' : 'transparent',
              color: unitSystem === 'imperial' ? '#ffffff' : 'var(--text-muted)',
              border: 'none',
              borderRadius: '8px',
              padding: '4px 12px',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
          >
            Imperial (ft/lbs)
          </button>
          <button
            onClick={() => setUnitSystem('metric')}
            style={{
              background: unitSystem === 'metric' ? '#f97316' : 'transparent',
              color: unitSystem === 'metric' ? '#ffffff' : 'var(--text-muted)',
              border: 'none',
              borderRadius: '8px',
              padding: '4px 12px',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
          >
            Metric (cm/kg)
          </button>
        </div>
      </div>

      {/* Archetype Quick Presets */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '0.8rem',
          fontWeight: 600,
          color: 'var(--text-muted)',
          marginBottom: '10px'
        }}>
          <Sparkles size={14} color="#f97316" />
          <span>Quick Archetype Presets</span>
        </div>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          {archetypes && archetypes.map((arch) => (
            <button
              key={arch.id}
              onClick={() => handleArchetypeClick(arch)}
              className={`preset-chip ${activeArchetypeId === arch.id ? 'active' : ''}`}
              title={arch.description}
            >
              <Zap size={12} />
              <span>{arch.title}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Form Controls Grid */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Height Slider */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.875rem', fontWeight: 600 }}>
              <Ruler size={16} color="#38bdf8" />
              <span>Height</span>
            </label>
            <span className="font-mono-data" style={{
              fontSize: '0.95rem',
              fontWeight: 700,
              color: '#38bdf8',
              background: 'rgba(56, 189, 248, 0.1)',
              padding: '2px 10px',
              borderRadius: '6px'
            }}>
              {unitSystem === 'imperial' ? `${ftIn.str} (${height_cm} cm)` : `${height_cm} cm (${ftIn.str})`}
            </span>
          </div>
          <input
            type="range"
            min="170"
            max="226"
            step="1"
            value={height_cm}
            onChange={(e) => handleHeightChange(e.target.value)}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '4px' }}>
            <span>5'7" (170cm)</span>
            <span>6'6" (198cm)</span>
            <span>7'5" (226cm)</span>
          </div>
        </div>

        {/* Weight Slider */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.875rem', fontWeight: 600 }}>
              <Weight size={16} color="#a855f7" />
              <span>Weight</span>
            </label>
            <span className="font-mono-data" style={{
              fontSize: '0.95rem',
              fontWeight: 700,
              color: '#a855f7',
              background: 'rgba(168, 85, 247, 0.1)',
              padding: '2px 10px',
              borderRadius: '6px'
            }}>
              {unitSystem === 'imperial' ? `${lbs} lbs (${weight_kg} kg)` : `${weight_kg} kg (${lbs} lbs)`}
            </span>
          </div>
          <input
            type="range"
            min="70"
            max="140"
            step="1"
            value={weight_kg}
            onChange={(e) => handleWeightChange(e.target.value)}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '4px' }}>
            <span>154 lbs (70kg)</span>
            <span>220 lbs (100kg)</span>
            <span>308 lbs (140kg)</span>
          </div>
        </div>

        {/* Games Played & Total Minutes Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '16px'
        }}>
          {/* Games Played */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', fontWeight: 600 }}>
                <Calendar size={15} color="#10b981" />
                <span>Games Played</span>
              </label>
              <span className="font-mono-data" style={{ fontSize: '0.9rem', fontWeight: 700, color: '#10b981' }}>
                {gp} GP
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="82"
              step="1"
              value={gp}
              onChange={(e) => handleGpChange(e.target.value)}
            />
          </div>

          {/* Total Minutes Played */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', fontWeight: 600 }}>
                <Clock size={15} color="#f59e0b" />
                <span>Season Minutes</span>
              </label>
              <span className="font-mono-data" style={{ fontSize: '0.9rem', fontWeight: 700, color: '#f59e0b' }}>
                {min} Min
              </span>
            </div>
            <input
              type="range"
              min="50"
              max="3100"
              step="25"
              value={min}
              onChange={(e) => handleMinChange(e.target.value)}
            />
          </div>
        </div>

        {/* Minutes per game calculated pill */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px dashed var(--border-color)',
          borderRadius: '10px',
          padding: '10px 14px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Projected Playing Time:</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="font-mono-data" style={{
              fontWeight: 800,
              fontSize: '1rem',
              color: Number(minPerGame) > 34 ? '#f97316' : Number(minPerGame) > 25 ? '#38bdf8' : '#94a3b8'
            }}>
              {minPerGame} MPG
            </span>
            <span style={{
              fontSize: '0.7rem',
              padding: '2px 6px',
              borderRadius: '4px',
              background: 'rgba(255, 255, 255, 0.06)',
              color: 'var(--text-muted)'
            }}>
              {Number(minPerGame) > 34 ? 'Star Starter' : Number(minPerGame) > 24 ? 'Regular Starter' : Number(minPerGame) > 15 ? '6th Man / Rotation' : 'Bench Depth'}
            </span>
          </div>
        </div>

        {/* Team & Position Dropdowns */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '16px'
        }}>
          {/* Team Dropdown */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
              NBA Team
            </label>
            <div style={{ position: 'relative' }}>
              <select
                className="form-input"
                value={team}
                onChange={(e) => {
                  setActiveArchetypeId(null);
                  setFormState(prev => ({ ...prev, team: e.target.value }));
                }}
                style={{
                  borderLeft: `4px solid ${selectedTeamInfo.primary || '#f97316'}`,
                  fontWeight: 600
                }}
              >
                {teams && teams.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Position Dropdown */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
              Primary Position
            </label>
            <select
              className="form-input"
              value={position}
              onChange={(e) => {
                setActiveArchetypeId(null);
                setFormState(prev => ({ ...prev, position: e.target.value }));
              }}
              style={{ fontWeight: 600 }}
            >
              {positions && positions.map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Prediction Trigger Button */}
        <button
          onClick={onPredict}
          disabled={isLoading}
          className="btn-primary"
          style={{ width: '100%', marginTop: '6px' }}
        >
          {isLoading ? (
            <>
              <RefreshCw size={18} className="animate-spin" />
              <span>Calculating Model Projections...</span>
            </>
          ) : (
            <>
              <Sparkles size={18} />
              <span>Generate NBA Projections</span>
              <ArrowRight size={18} />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
