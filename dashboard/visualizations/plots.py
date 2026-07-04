# dashboard/visualizations/plots.py
import numpy as np
import plotly.graph_objects as go

def render_tactical_map(lats, lons, risk_timeline, map_style, df_perturbed=None, show_cf=False):
    fig = go.Figure()
    
    aligned_timeline = None
    hover_label = "Pointwise Feature Vector Tracking"
    
    if risk_timeline is not None:
        risk_arr = np.atleast_1d(risk_timeline)
        if len(risk_arr) > 0:
            if len(risk_arr) == len(lats):
                aligned_timeline = risk_arr
                hover_label = "Pointwise Model Risk Estimate"
            elif len(risk_arr) == 1:
                aligned_timeline = np.full(len(lats), risk_arr[0])
                hover_label = "Static Window Sequence Risk"
            else:
                # Continuous dimensional mapping safely scales metrics across coordinates matrix
                aligned_timeline = np.interp(
                    np.linspace(0, len(risk_arr) - 1, len(lats)),
                    np.arange(len(risk_arr)),
                    risk_arr
                )
                hover_label = "Interpolated Window Risk Projection"

    fig.add_trace(go.Scattermap(
        lon=lons, lat=lats, mode="markers+lines",
        marker=dict(
            size=7, 
            color=aligned_timeline if aligned_timeline is not None else '#3B82F6',
            colorscale=[[0, '#10B981'], [0.5, '#F59E0B'], [1, '#EF4444']] if aligned_timeline is not None else None,
            cmin=0.0, cmax=1.0, showscale=True if aligned_timeline is not None else False,
            colorbar=dict(title="Risk Horizon Scale", thickness=12, len=0.4, y=0.8)
        ),
        line=dict(color="rgba(128,128,128,0.4)", width=3),
        name="Observed Track",
        hovertemplate="<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<br>Risk Score: %{marker.color:.4f}<extra></extra>",
        text=[hover_label] * len(lats)
    ))

    if show_cf and df_perturbed is not None:
        fig.add_trace(go.Scattermap(
            lat=df_perturbed["lat_raw"].to_numpy(), lon=df_perturbed["lon_raw"].to_numpy(),
            mode="lines+markers", line=dict(width=3, color="#6366F1"),
            name="Counterfactual Hypothesis Path"
        ))
    
    fig.add_trace(go.Scattermap(lon=[lons[0]], lat=[lats[0]], mode="markers", marker=dict(size=12, color="#38BDF8"), name="Origin"))
    fig.add_trace(go.Scattermap(lon=[lons[-1]], lat=[lats[-1]], mode="markers", marker=dict(size=16, color="#EF4444"), name="Vessel Head"))

    lat_range = np.max(lats) - np.min(lats) if len(lats) > 1 else 0.1
    calculated_zoom = max(1, min(12, int(11.5 - np.log2(max(lat_range, 0.01)))))

    fig.update_layout(
        map=dict(style=map_style, center=dict(lat=float(np.mean(lats)), lon=float(np.mean(lons))), zoom=calculated_zoom),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=580, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(127,127,127,0.1)")
    )
    return fig

def render_risk_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": "#3B82F6", "thickness": 0.25},
            "steps": [{"range": [0, 0.25], "color": "rgba(16,185,129,0.20)"}, {"range": [0.25, 0.5], "color": "rgba(16,185,129,0.35)"}, {"range": [0.5, 0.75], "color": "rgba(245,158,11,0.35)"}, {"range": [0.75, 1], "color": "rgba(239,68,68,0.50)"}]
        }
    ))
    fig.update_layout(height=150, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=20, b=10))
    return fig