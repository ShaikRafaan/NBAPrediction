import React from 'react';
import { Activity, Sparkles, Database, ShieldCheck, Flame } from 'lucide-react';

export default function Navbar({ isOnline, totalPlayers, onReset }) {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-color)',
      background: 'rgba(10, 14, 23, 0.85)',
      backdropFilter: 'blur(20px)',
      position: 'sticky',
      top: 0,
      zIndex: 50
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        {/* Logo and Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '44px',
            height: '44px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '22px',
            boxShadow: '0 4px 16px rgba(249, 115, 22, 0.4)'
          }}>
            🏀
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{
                fontSize: '1.25rem',
                fontWeight: 800,
                letterSpacing: '-0.02em',
                color: '#ffffff',
                margin: 0
              }}>
                NBA PROJECTION ENGINE
              </h1>
              <span style={{
                background: 'rgba(249, 115, 22, 0.15)',
                border: '1px solid rgba(249, 115, 22, 0.3)',
                color: '#fb923c',
                fontSize: '0.7rem',
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: '6px',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                v2.0 AI
              </span>
            </div>
            <p style={{
              fontSize: '0.8rem',
              color: 'var(--text-muted)',
              margin: 0
            }}>
              Machine Learning Player Scouting & Stat Forecaster
            </p>
          </div>
        </div>

        {/* Right Info Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: '10px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-color)',
            fontSize: '0.8rem'
          }}>
            <Database size={14} color="#38bdf8" />
            <span style={{ color: 'var(--text-muted)' }}>Dataset:</span>
            <strong style={{ color: '#ffffff' }}>{totalPlayers || '500+'} Active Players</strong>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            borderRadius: '10px',
            background: isOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: isOnline ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)',
            fontSize: '0.75rem',
            fontWeight: 600,
            color: isOnline ? '#34d399' : '#f87171'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isOnline ? '#10b981' : '#ef4444',
              boxShadow: isOnline ? '0 0 8px #10b981' : 'none'
            }} />
            {isOnline ? 'Model Active (XGBoost)' : 'Backend Connecting...'}
          </div>
        </div>
      </div>
    </header>
  );
}
