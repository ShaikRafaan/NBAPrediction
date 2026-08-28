import React from 'react';
import { Target, Activity } from 'lucide-react';

export default function RadarChart({ radarData }) {
  if (!radarData || radarData.length === 0) return null;

  const size = 260;
  const center = size / 2;
  const radius = size * 0.38;
  const totalSides = radarData.length;
  const angleStep = (Math.PI * 2) / totalSides;

  // Compute polygon points for the dataset
  const points = radarData.map((item, index) => {
    const angle = index * angleStep - Math.PI / 2;
    const r = (Math.min(100, Math.max(0, item.value)) / 100) * radius;
    const x = center + r * Math.cos(angle);
    const y = center + r * Math.sin(angle);
    return `${x},${y}`;
  }).join(' ');

  // Compute background concentric web polygons (25%, 50%, 75%, 100%)
  const levels = [0.25, 0.5, 0.75, 1.0];

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Target size={18} color="#f97316" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
            Playstyle Archetype Matrix
          </h3>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          0-100 Skill Index
        </span>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(200px, 280px) 1fr',
        gap: '20px',
        alignItems: 'center'
      }}>
        {/* SVG Radar Chart */}
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <svg width={size} height={size} style={{ overflow: 'visible' }}>
            {/* Background Webs */}
            {levels.map((lvl, lIdx) => {
              const lvlPoints = radarData.map((_, index) => {
                const angle = index * angleStep - Math.PI / 2;
                const r = radius * lvl;
                const x = center + r * Math.cos(angle);
                const y = center + r * Math.sin(angle);
                return `${x},${y}`;
              }).join(' ');

              return (
                <polygon
                  key={lIdx}
                  points={lvlPoints}
                  fill={lIdx === levels.length - 1 ? 'rgba(255, 255, 255, 0.02)' : 'none'}
                  stroke="rgba(255, 255, 255, 0.08)"
                  strokeWidth="1"
                />
              );
            })}

            {/* Axis lines */}
            {radarData.map((_, index) => {
              const angle = index * angleStep - Math.PI / 2;
              const x2 = center + radius * Math.cos(angle);
              const y2 = center + radius * Math.sin(angle);
              return (
                <line
                  key={index}
                  x1={center}
                  y1={center}
                  x2={x2}
                  y2={y2}
                  stroke="rgba(255, 255, 255, 0.08)"
                  strokeWidth="1"
                />
              );
            })}

            {/* Data Polygon Fill & Stroke */}
            <polygon
              points={points}
              fill="rgba(249, 115, 22, 0.25)"
              stroke="#f97316"
              strokeWidth="2.5"
              style={{
                filter: 'drop-shadow(0 0 8px rgba(249, 115, 22, 0.5))',
                transition: 'all 0.4s ease'
              }}
            />

            {/* Vertex Dots & Category Labels */}
            {radarData.map((item, index) => {
              const angle = index * angleStep - Math.PI / 2;
              const r = (Math.min(100, Math.max(0, item.value)) / 100) * radius;
              const x = center + r * Math.cos(angle);
              const y = center + r * Math.sin(angle);

              // Label pos
              const labelR = radius + 22;
              const lx = center + labelR * Math.cos(angle);
              const ly = center + labelR * Math.sin(angle);

              return (
                <g key={index}>
                  <circle
                    cx={x}
                    cy={y}
                    r="4"
                    fill="#fb923c"
                    stroke="#ffffff"
                    strokeWidth="1.5"
                  />
                  <text
                    x={lx}
                    y={ly + 4}
                    fill="#94a3b8"
                    fontSize="11"
                    fontWeight="600"
                    textAnchor="middle"
                    style={{ userSelect: 'none' }}
                  >
                    {item.category}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Breakdown Progress Bars */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {radarData.map((item, idx) => (
            <div key={idx}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '0.8rem',
                marginBottom: '4px',
                fontWeight: 600
              }}>
                <span style={{ color: 'var(--text-main)' }}>{item.category}</span>
                <span className="font-mono-data" style={{ color: '#fb923c', fontWeight: 700 }}>
                  {item.value} / 100
                </span>
              </div>
              <div style={{
                height: '6px',
                background: 'rgba(255, 255, 255, 0.06)',
                borderRadius: '3px',
                overflow: 'hidden'
              }}>
                <div style={{
                  height: '100%',
                  width: `${item.value}%`,
                  background: 'linear-gradient(90deg, #f97316 0%, #38bdf8 100%)',
                  borderRadius: '3px',
                  transition: 'width 0.5s cubic-bezier(0.16, 1, 0.3, 1)'
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
