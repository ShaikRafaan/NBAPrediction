import React from 'react';
import { Award, Flame, Target, TrendingUp, ShieldAlert, BarChart3 } from 'lucide-react';
import { NBA_TEAMS_INFO } from '../utils/nbaData';

export default function StatHero({ prediction, formState }) {
  if (!prediction) return null;

  const { per_game, totals, percentiles, composite } = prediction;
  const teamInfo = NBA_TEAMS_INFO[formState.team] || { primary: '#f97316', secondary: '#ffffff', abbr: formState.team.substring(0, 3).toUpperCase(), city: '' };

  const getPercentileColor = (pct) => {
    if (pct >= 90) return { bg: 'rgba(234, 88, 12, 0.2)', border: 'rgba(234, 88, 12, 0.4)', text: '#fb923c', label: `Top ${(100 - pct).toFixed(0)}%` };
    if (pct >= 75) return { bg: 'rgba(16, 185, 129, 0.15)', border: 'rgba(16, 185, 129, 0.3)', text: '#34d399', label: `${pct.toFixed(0)}th %ile` };
    if (pct >= 50) return { bg: 'rgba(56, 189, 248, 0.15)', border: 'rgba(56, 189, 248, 0.3)', text: '#38bdf8', label: `${pct.toFixed(0)}th %ile` };
    return { bg: 'rgba(148, 163, 184, 0.1)', border: 'rgba(148, 163, 184, 0.2)', text: '#94a3b8', label: `${pct.toFixed(0)}th %ile` };
  };

  const statCards = [
    {
      label: 'Points',
      short: 'PTS',
      value: per_game.PTS.toFixed(1),
      total: totals.PTS.toFixed(0),
      percentile: percentiles.league.PTS,
      color: '#f97316',
      subtext: 'Per Game'
    },
    {
      label: 'Assists',
      short: 'AST',
      value: per_game.AST.toFixed(1),
      total: totals.AST.toFixed(0),
      percentile: percentiles.league.AST,
      color: '#38bdf8',
      subtext: 'Per Game'
    },
    {
      label: 'Rebounds',
      short: 'REB',
      value: per_game.REB.toFixed(1),
      total: totals.REB.toFixed(0),
      percentile: percentiles.league.REB,
      color: '#10b981',
      subtext: 'Per Game'
    },
    {
      label: 'Field Goal %',
      short: 'FG%',
      value: (per_game['FG%'] * 100).toFixed(1) + '%',
      total: `${totals.FGM.toFixed(0)} / ${totals.FGA.toFixed(0)} FGM`,
      percentile: percentiles.league['FG%'],
      color: '#a855f7',
      subtext: 'Efficiency'
    },
    {
      label: 'FG Made',
      short: 'FGM',
      value: per_game.FGM.toFixed(1),
      total: `${totals.FGM.toFixed(0)} Makes`,
      percentile: percentiles.league.PTS,
      color: '#eab308',
      subtext: 'Per Game'
    },
    {
      label: 'FG Attempted',
      short: 'FGA',
      value: per_game.FGA.toFixed(1),
      total: `${totals.FGA.toFixed(0)} Shots`,
      percentile: percentiles.league.PTS,
      color: '#ec4899',
      subtext: 'Per Game'
    }
  ];

  return (
    <div className="glass-panel animate-fade-in" style={{ padding: '24px', position: 'relative', overflow: 'hidden' }}>
      
      {/* Background Court Branding Glow */}
      <div style={{
        position: 'absolute',
        top: '-40px',
        right: '-40px',
        width: '200px',
        height: '200px',
        borderRadius: '50%',
        background: `radial-gradient(circle, ${teamInfo.primary}33 0%, transparent 70%)`,
        pointerEvents: 'none',
        zIndex: 0
      }} />

      {/* Header Profile Bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        paddingBottom: '20px',
        borderBottom: '1px solid var(--border-color)',
        position: 'relative',
        zIndex: 1
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Team Logo / Badge */}
          <div style={{
            width: '56px',
            height: '56px',
            borderRadius: '14px',
            background: `linear-gradient(135deg, ${teamInfo.primary} 0%, #111827 100%)`,
            border: `2px solid ${teamInfo.primary}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '18px',
            fontWeight: 900,
            color: '#ffffff',
            boxShadow: `0 4px 20px ${teamInfo.primary}44`
          }}>
            {teamInfo.abbr}
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: 0, color: '#ffffff' }}>
                {teamInfo.city ? `${teamInfo.city} ` : ''}{formState.team} Projected {formState.position}
              </h2>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '4px' }}>
              <span style={{
                background: 'rgba(255, 255, 255, 0.06)',
                padding: '2px 8px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                color: 'var(--text-muted)'
              }}>
                {formState.gp} Games • {formState.min} Total Mins ({per_game.MIN} MPG)
              </span>
              <span style={{
                background: 'rgba(255, 255, 255, 0.06)',
                padding: '2px 8px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                color: 'var(--text-muted)'
              }}>
                {formState.height_cm} cm • {formState.weight_kg} kg
              </span>
            </div>
          </div>
        </div>

        {/* Tier & Overall Rating */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
          background: 'rgba(15, 23, 42, 0.7)',
          border: '1px solid var(--border-color)',
          padding: '8px 16px',
          borderRadius: '12px'
        }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em' }}>
              Impact Tier
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 800, color: '#fb923c' }}>
              {composite.tier}
            </div>
          </div>
          <div style={{
            width: '44px',
            height: '44px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.2rem',
            fontWeight: 900,
            color: '#ffffff',
            boxShadow: '0 4px 14px rgba(249, 115, 22, 0.4)'
          }}>
            {Math.round(composite.score)}
          </div>
        </div>
      </div>

      {/* Primary Stat Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
        gap: '14px',
        marginTop: '20px',
        position: 'relative',
        zIndex: 1
      }}>
        {statCards.map((card, idx) => {
          const pctStyle = getPercentileColor(card.percentile);
          return (
            <div
              key={idx}
              style={{
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid var(--border-color)',
                borderRadius: '14px',
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                transition: 'all 0.2s ease',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              {/* Stat Top Accent Line */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '3px',
                background: card.color
              }} />

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    {card.short}
                  </span>
                  <span style={{
                    fontSize: '0.65rem',
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: pctStyle.bg,
                    border: `1px solid ${pctStyle.border}`,
                    color: pctStyle.text
                  }}>
                    {pctStyle.label}
                  </span>
                </div>

                <div className="font-mono-data" style={{
                  fontSize: '1.9rem',
                  fontWeight: 900,
                  color: '#ffffff',
                  lineHeight: 1.1,
                  letterSpacing: '-0.03em'
                }}>
                  {card.value}
                </div>
              </div>

              <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid rgba(255, 255, 255, 0.05)', fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                <span>Season: {card.total}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
