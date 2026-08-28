import React from 'react';
import { Table, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

export default function LeagueContextTable({ prediction, position, scatterData }) {
  if (!prediction || !scatterData) return null;

  const { per_game } = prediction;

  // Compute position and league averages from scatterData
  const leaguePlayers = scatterData;
  const posPlayers = scatterData.filter(p => p.position === position || p.position.includes(position.split('-')[0]));

  const calcAvg = (arr, key) => {
    if (!arr || arr.length === 0) return 0;
    const sum = arr.reduce((acc, p) => acc + (p[key] || 0), 0);
    return sum / arr.length;
  };

  const rows = [
    {
      metric: 'Points (PTS)',
      proj: per_game.PTS,
      posAvg: calcAvg(posPlayers, 'pts_pg'),
      leagueAvg: calcAvg(leaguePlayers, 'pts_pg'),
      format: (v) => v.toFixed(1)
    },
    {
      metric: 'Assists (AST)',
      proj: per_game.AST,
      posAvg: calcAvg(posPlayers, 'ast_pg'),
      leagueAvg: calcAvg(leaguePlayers, 'ast_pg'),
      format: (v) => v.toFixed(1)
    },
    {
      metric: 'Rebounds (REB)',
      proj: per_game.REB,
      posAvg: calcAvg(posPlayers, 'reb_pg'),
      leagueAvg: calcAvg(leaguePlayers, 'reb_pg'),
      format: (v) => v.toFixed(1)
    },
    {
      metric: 'Field Goal %',
      proj: per_game['FG%'] * 100,
      posAvg: calcAvg(posPlayers, 'fg_pct') * 100,
      leagueAvg: calcAvg(leaguePlayers, 'fg_pct') * 100,
      format: (v) => v.toFixed(1) + '%'
    },
    {
      metric: 'Minutes / Game',
      proj: per_game.MIN,
      posAvg: calcAvg(posPlayers, 'min_pg'),
      leagueAvg: calcAvg(leaguePlayers, 'min_pg'),
      format: (v) => v.toFixed(1)
    }
  ];

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Table size={18} color="#10b981" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
            Benchmark vs League & Position Average
          </h3>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Position: {position}
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-dim)' }}>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Metric</th>
              <th style={{ padding: '10px 12px', fontWeight: 700, color: '#f97316' }}>Projected</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Pos Avg ({posPlayers.length})</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>League Avg</th>
              <th style={{ padding: '10px 12px', fontWeight: 600, textAlign: 'right' }}>vs League Diff</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => {
              const diff = row.proj - row.leagueAvg;
              const isPositive = diff > 0.05;
              const isNegative = diff < -0.05;

              return (
                <tr
                  key={idx}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                    transition: 'background 0.15s ease'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '12px', fontWeight: 600, color: '#ffffff' }}>
                    {row.metric}
                  </td>
                  <td className="font-mono-data" style={{ padding: '12px', fontWeight: 800, color: '#f97316', fontSize: '0.95rem' }}>
                    {row.format(row.proj)}
                  </td>
                  <td className="font-mono-data" style={{ padding: '12px', color: 'var(--text-muted)' }}>
                    {row.format(row.posAvg)}
                  </td>
                  <td className="font-mono-data" style={{ padding: '12px', color: 'var(--text-muted)' }}>
                    {row.format(row.leagueAvg)}
                  </td>
                  <td style={{ padding: '12px', textAlign: 'right' }}>
                    <span style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      padding: '2px 8px',
                      borderRadius: '6px',
                      background: isPositive ? 'rgba(16, 185, 129, 0.15)' : isNegative ? 'rgba(239, 68, 68, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                      color: isPositive ? '#34d399' : isNegative ? '#f87171' : 'var(--text-dim)'
                    }}>
                      {isPositive ? <ArrowUpRight size={13} /> : isNegative ? <ArrowDownRight size={13} /> : <Minus size={13} />}
                      {diff > 0 ? `+${row.format(diff)}` : row.format(diff)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
