import React, { useState, useEffect, useRef } from 'react';
import { ScatterChart as ScatterIcon, Filter, Info, Eye } from 'lucide-react';

export default function ScatterChart({ predictedStat, formState, scatterData, position }) {
  const [activeStat, setActiveStat] = useState('pts_pg');
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [filterSamePos, setFilterSamePos] = useState(true);
  const canvasRef = useRef(null);

  const statOptions = [
    { key: 'pts_pg', label: 'Points / Game', unit: 'PTS', predKey: 'PTS' },
    { key: 'ast_pg', label: 'Assists / Game', unit: 'AST', predKey: 'AST' },
    { key: 'reb_pg', label: 'Rebounds / Game', unit: 'REB', predKey: 'REB' },
    { key: 'fg_pct', label: 'Field Goal %', unit: 'FG%', predKey: 'FG%' },
    { key: 'fgm_pg', label: 'Field Goals Made / Game', unit: 'FGM', predKey: 'FGM' },
    { key: 'fga_pg', label: 'Field Goals Attempted / Game', unit: 'FGA', predKey: 'FGA' }
  ];

  const currentStatConfig = statOptions.find(s => s.key === activeStat) || statOptions[0];

  // Filter dataset points
  const points = (scatterData || []).filter(p => {
    if (!filterSamePos) return true;
    return p.position === position || p.position.includes(position.split('-')[0]);
  });

  const predictedPoint = {
    name: '🎯 Projected Player',
    team: formState.team,
    position: formState.position,
    min: Number(formState.min),
    val: predictedStat?.per_game?.[currentStatConfig.predKey] || 0
  };

  // Canvas bounds & rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 30, right: 30, bottom: 45, left: 50 };

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Calculate ranges
    const xValues = points.map(p => p.min).concat([predictedPoint.min]);
    const yValues = points.map(p => p[activeStat]).concat([predictedPoint.val]);

    const minX = 0;
    const maxX = Math.max(3200, Math.max(...xValues) * 1.05);

    const minY = 0;
    const maxY = Math.max(1, Math.max(...yValues) * 1.15);

    const scaleX = (val) => padding.left + ((val - minX) / (maxX - minX)) * (width - padding.left - padding.right);
    const scaleY = (val) => height - padding.bottom - ((val - minY) / (maxY - minY)) * (height - padding.top - padding.bottom);

    // Draw Grid & Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;

    // Y Grid lines (4 ticks)
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const val = minY + (i / yTicks) * (maxY - minY);
      const y = scaleY(val);

      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px Outfit, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const label = activeStat === 'fg_pct' ? `${(val * 100).toFixed(0)}%` : val.toFixed(1);
      ctx.fillText(label, padding.left - 10, y);
    }

    // X Grid lines (5 ticks)
    const xTicks = 5;
    for (let i = 0; i <= xTicks; i++) {
      const val = minX + (i / xTicks) * (maxX - minX);
      const x = scaleX(val);

      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`${Math.round(val)}m`, x, height - padding.bottom + 8);
    }

    // Axis Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '600 11px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Total Season Minutes Played', (width + padding.left - padding.right) / 2, height - 12);

    // Render NBA peer points
    points.forEach((p) => {
      const px = scaleX(p.min);
      const py = scaleY(p[activeStat]);

      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.45)';
      ctx.fill();
    });

    // Render Predicted Player Target (Glow & Crosshair)
    const targetX = scaleX(predictedPoint.min);
    const targetY = scaleY(predictedPoint.val);

    // Crosshairs
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(249, 115, 22, 0.4)';
    ctx.beginPath();
    ctx.moveTo(padding.left, targetY);
    ctx.lineTo(targetX, targetY);
    ctx.lineTo(targetX, height - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Glow circle
    const glowGradient = ctx.createRadialGradient(targetX, targetY, 2, targetX, targetY, 18);
    glowGradient.addColorStop(0, 'rgba(249, 115, 22, 0.9)');
    glowGradient.addColorStop(0.5, 'rgba(249, 115, 22, 0.3)');
    glowGradient.addColorStop(1, 'rgba(249, 115, 22, 0)');
    ctx.fillStyle = glowGradient;
    ctx.beginPath();
    ctx.arc(targetX, targetY, 18, 0, Math.PI * 2);
    ctx.fill();

    // Solid Target Dot
    ctx.beginPath();
    ctx.arc(targetX, targetY, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#f97316';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.stroke();

  }, [points, activeStat, predictedStat, formState, filterSamePos]);

  // Handle Mouse Hover on Canvas for Interactive Tooltips
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const padding = { top: 30, right: 30, bottom: 45, left: 50 };
    const width = rect.width;
    const height = rect.height;

    const xValues = points.map(p => p.min).concat([predictedPoint.min]);
    const yValues = points.map(p => p[activeStat]).concat([predictedPoint.val]);

    const minX = 0;
    const maxX = Math.max(3200, Math.max(...xValues) * 1.05);
    const minY = 0;
    const maxY = Math.max(1, Math.max(...yValues) * 1.15);

    const scaleX = (val) => padding.left + ((val - minX) / (maxX - minX)) * (width - padding.left - padding.right);
    const scaleY = (val) => height - padding.bottom - ((val - minY) / (maxY - minY)) * (height - padding.top - padding.bottom);

    // Check predicted point first
    const predX = scaleX(predictedPoint.min);
    const predY = scaleY(predictedPoint.val);
    const predDist = Math.hypot(mouseX - predX, mouseY - predY);

    if (predDist < 14) {
      setHoveredPoint({
        name: '🎯 Projected Player (Your Build)',
        team: formState.team,
        position: formState.position,
        min: formState.min,
        val: predictedPoint.val,
        x: predX,
        y: predY,
        isPredicted: true
      });
      return;
    }

    // Check peer points
    let closest = null;
    let closestDist = 12;

    points.forEach(p => {
      const px = scaleX(p.min);
      const py = scaleY(p[activeStat]);
      const dist = Math.hypot(mouseX - px, mouseY - py);
      if (dist < closestDist) {
        closestDist = dist;
        closest = {
          name: p.name,
          team: p.team,
          position: p.position,
          min: p.min,
          val: p[activeStat],
          x: px,
          y: py,
          isPredicted: false
        };
      }
    });

    setHoveredPoint(closest);
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      {/* Chart Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
        marginBottom: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ScatterIcon size={18} color="#f97316" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
            Interactive Position Distribution & League Scatter
          </h3>
        </div>

        {/* Position Filter Toggle */}
        <button
          onClick={() => setFilterSamePos(!filterSamePos)}
          className="btn-secondary"
          style={{
            fontSize: '0.75rem',
            background: filterSamePos ? 'rgba(249, 115, 22, 0.15)' : 'rgba(255, 255, 255, 0.05)',
            borderColor: filterSamePos ? 'rgba(249, 115, 22, 0.4)' : 'var(--border-color)',
            color: filterSamePos ? '#fb923c' : 'var(--text-muted)'
          }}
        >
          <Filter size={12} />
          <span>{filterSamePos ? `Filtering: ${position}s Only` : 'Showing All Positions'}</span>
        </button>
      </div>

      {/* Metric Selector Tabs */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '8px',
        marginBottom: '16px'
      }}>
        {statOptions.map(stat => (
          <button
            key={stat.key}
            onClick={() => setActiveStat(stat.key)}
            className={`preset-chip ${activeStat === stat.key ? 'active' : ''}`}
            style={{ fontSize: '0.75rem' }}
          >
            <span>{stat.label}</span>
          </button>
        ))}
      </div>

      {/* Canvas Area */}
      <div style={{ position: 'relative', width: '100%', height: '320px', background: '#090d16', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ width: '100%', height: '100%', display: 'block', cursor: 'crosshair' }}
        />

        {/* Tooltip Overlay */}
        {hoveredPoint && (
          <div style={{
            position: 'absolute',
            left: `${Math.min(hoveredPoint.x + 12, canvasRef.current?.getBoundingClientRect().width - 180 || 200)}px`,
            top: `${Math.max(10, hoveredPoint.y - 45)}px`,
            background: 'rgba(10, 14, 23, 0.95)',
            border: hoveredPoint.isPredicted ? '1px solid #f97316' : '1px solid #38bdf8',
            borderRadius: '8px',
            padding: '8px 12px',
            pointerEvents: 'none',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
            zIndex: 10
          }}>
            <div style={{ fontWeight: 700, fontSize: '0.85rem', color: hoveredPoint.isPredicted ? '#fb923c' : '#ffffff' }}>
              {hoveredPoint.name}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {hoveredPoint.team} • {hoveredPoint.position}
            </div>
            <div className="font-mono-data" style={{ fontSize: '0.85rem', fontWeight: 800, color: hoveredPoint.isPredicted ? '#f97316' : '#38bdf8', marginTop: '4px' }}>
              {activeStat === 'fg_pct' ? `${(hoveredPoint.val * 100).toFixed(1)}%` : hoveredPoint.val.toFixed(1)} {currentStatConfig.unit} ({hoveredPoint.min} min)
            </div>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '12px', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#f97316', border: '2px solid #fff', display: 'inline-block' }} />
            Projected Player
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#38bdf8', opacity: 0.6, display: 'inline-block' }} />
            Active NBA Players ({points.length})
          </span>
        </div>
        <span>Hover dots to inspect stats</span>
      </div>
    </div>
  );
}
