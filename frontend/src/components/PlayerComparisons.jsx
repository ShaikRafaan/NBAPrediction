import React from 'react';
import { Users, CheckCircle2, TrendingUp } from 'lucide-react';
import { NBA_TEAMS_INFO } from '../utils/nbaData';

export default function PlayerComparisons({ comparisons, projectedStats }) {
  if (!comparisons || comparisons.length === 0) return null;

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '18px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Users size={18} color="#38bdf8" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
            Closest NBA Player Comparisons
          </h3>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Multi-dimensional Vector Match
        </span>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '14px'
      }}>
        {comparisons.map((player, idx) => {
          const teamMeta = NBA_TEAMS_INFO[player.team] || { primary: '#38bdf8', abbr: player.team ? player.team.substring(0, 3).toUpperCase() : 'NBA' };

          return (
            <div
              key={idx}
              style={{
                background: 'rgba(15, 23, 42, 0.65)',
                border: '1px solid var(--border-color)',
                borderRadius: '14px',
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                transition: 'transform 0.2s ease, border-color 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.4)';
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-color)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              {/* Header */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                  <div>
                    <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff', margin: 0 }}>
                      {player.name}
                    </h4>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {player.team} • {player.position}
                    </div>
                  </div>

                  {/* Similarity Pill */}
                  <span style={{
                    fontSize: '0.75rem',
                    fontWeight: 800,
                    padding: '3px 8px',
                    borderRadius: '6px',
                    background: 'rgba(56, 189, 248, 0.15)',
                    border: '1px solid rgba(56, 189, 248, 0.3)',
                    color: '#38bdf8'
                  }}>
                    {player.similarity}% Match
                  </span>
                </div>

                {/* Stat Grid Comparison */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, 1fr)',
                  gap: '8px',
                  background: 'rgba(0, 0, 0, 0.25)',
                  padding: '10px',
                  borderRadius: '10px',
                  marginTop: '12px'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', fontWeight: 600 }}>PTS</div>
                    <div className="font-mono-data" style={{ fontSize: '0.9rem', fontWeight: 800, color: '#ffffff' }}>
                      {player.pts_pg.toFixed(1)}
                    </div>
                  </div>

                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', fontWeight: 600 }}>AST</div>
                    <div className="font-mono-data" style={{ fontSize: '0.9rem', fontWeight: 800, color: '#ffffff' }}>
                      {player.ast_pg.toFixed(1)}
                    </div>
                  </div>

                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', fontWeight: 600 }}>REB</div>
                    <div className="font-mono-data" style={{ fontSize: '0.9rem', fontWeight: 800, color: '#ffffff' }}>
                      {player.reb_pg.toFixed(1)}
                    </div>
                  </div>

                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', fontWeight: 600 }}>FG%</div>
                    <div className="font-mono-data" style={{ fontSize: '0.9rem', fontWeight: 800, color: '#ffffff' }}>
                      {(player.fg_pct * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '12px', fontSize: '0.72rem', color: 'var(--text-dim)', display: 'flex', justifyContent: 'space-between' }}>
                <span>{player.gp} GP Tracked</span>
                <span>{player.min_pg} MPG</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
