"""
Competitive Reconstruction: Visualizing Water-Filling in CSR
=============================================================
Interactive demonstration of how indicators compete for salience allocation.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="CSR Water-Filling Demo",
    page_icon="🌊",
    layout="wide"
)

# ============================================================================
# CORE ALGORITHM
# ============================================================================

def water_filling_step_by_step(y, n_steps=50, c_scale=1.0):
    """
    Compute salience via water-filling with step-by-step tracking.
    """
    J = len(y)
    eta = np.sum(np.maximum(y, 0.1))  # Volume estimate
    
    # Reconstruction drives
    d = eta * y
    # Curvature cost - heavily affects the "sharpness" of competition
    c = (eta ** 2) * c_scale
    
    # Range for threshold search
    nu_min = -np.max(d) - c
    nu_max = c - np.min(d) + c
    
    history = []
    
    # Bisection sweep
    for step in range(n_steps):
        nu = nu_min + (nu_max - nu_min) * step / (n_steps - 1)
        s_raw = np.maximum(0, (d + nu) / c)
        s_sum = np.sum(s_raw)
        
        if s_sum > 0:
            s_normalized = s_raw / s_sum
        else:
            s_normalized = np.ones(J) / J
            
        history.append({
            'step': step,
            'nu': nu,
            's_raw': s_raw.copy(),
            's_sum': s_sum,
            's_normalized': s_normalized.copy(),
            'active': (s_raw > 0).astype(int)
        })
    
    # Find best step
    sums = [h['s_sum'] for h in history]
    best_idx = np.argmin(np.abs(np.array(sums) - 1))
    
    return {
        'history': history,
        'best_idx': best_idx,
        'd': d,
        'c': c,
        'eta': eta,
        'y': y
    }

def compute_final_salience(y, c_scale=1.0):
    """Compute final precise salience."""
    J = len(y)
    eta = np.sum(np.maximum(y, 0.1))
    d = eta * y
    c = (eta ** 2) * c_scale
    
    def salience_sum(nu):
        return np.sum(np.maximum(0, (d + nu) / c))
    
    nu_low, nu_high = -np.max(d) - c*2, c*2
    for _ in range(100):
        nu_mid = (nu_low + nu_high) / 2
        if salience_sum(nu_mid) > 1:
            nu_high = nu_mid
        else:
            nu_low = nu_mid
    
    s = np.maximum(0, (d + nu_mid) / c)
    if np.sum(s) > 0:
        s = s / np.sum(s)
    else:
        s = np.ones(J)/J
    return s, nu_mid, eta

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🌊 Competitive Salience-Reconstruction")
st.markdown("### How Indicators Compete for Measurement Attention")

# --- Sidebar Inputs ---
st.sidebar.header("📊 Response Profile")

# Callbacks
def set_uniform():
    for i in range(5):
        st.session_state[f"item_{i}"] = 3

def set_skewed():
    vals = [5, 5, 2, 1, 1]
    for i, v in enumerate(vals):
        st.session_state[f"item_{i}"] = v

# Sliders
default_responses = [5, 4, 2, 1, 3]
item_labels = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]
responses = []
for i, (label, default) in enumerate(zip(item_labels, default_responses)):
    val = st.sidebar.slider(label, min_value=1, max_value=5, value=default, key=f"item_{i}")
    responses.append(val)
y_input = np.array(responses, dtype=float)

st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
col1.button("Uniform", on_click=set_uniform)
col2.button("Skewed", on_click=set_skewed)

# --- THE TOGGLE ---
st.sidebar.markdown("---")
break_equiv = st.sidebar.checkbox(
    "Break Equivalence",
    value=False,
    help="Adds noise and increases competition to demonstrate why Salience != Normalization."
)

if break_equiv:
    st.sidebar.warning("⚡ **High-Competition Mode**")
    st.sidebar.markdown("""
    * **Noise Added**: Simulating real-world imperfection.
    * **Cost Lowered**: Competition is sharper.
    * **Result**: Weak items drop to zero (Sparsity).
    """)
    np.random.seed(42)
    noise = np.random.normal(0, 0.4, size=y_input.shape)
    y_effective = np.maximum(0.1, y_input + noise)
    c_factor = 0.2  # Aggressive competition
else:
    y_effective = y_input
    c_factor = 1.0  # Standard

# Calculations
results = water_filling_step_by_step(y_effective, c_scale=c_factor)
final_s, final_nu, eta = compute_final_salience(y_effective, c_scale=c_factor)
simple_norm = y_effective / np.sum(y_effective)

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🎬 Animation", "📐 The Math", "🔬 Deep Dive"])

with tab1:
    st.markdown("## Watch Indicators Compete")
    
    # Slider
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        step = st.slider(
            "Water Level (Threshold ν)", 
            min_value=0, 
            max_value=len(results['history'])-1, 
            value=results['best_idx']
        )
    current = results['history'][step]
    
    # --- PLOT SETUP ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Response Profile (Input)",
            "Salience vs. Normalization (Output)",
            "Competition Dynamics",
            "Who's Active?"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 1. TOP LEFT: INPUT (Bars)
    fig.add_trace(
        go.Bar(x=item_labels, y=y_effective, marker_color=colors, name="Response"),
        row=1, col=1
    )
    
    # 2. TOP RIGHT: OUTPUT (Area + Line Comparison)
    # This solves the "It looks the same" problem.
    
    # A) Simple Normalization (Baseline)
    fig.add_trace(
        go.Scatter(
            x=item_labels, 
            y=simple_norm,
            mode='lines+markers',
            line=dict(color='gray', width=2, dash='dot'),
            marker=dict(size=8, symbol='circle-open'),
            name="Simple Norm (Proportional)"
        ),
        row=1, col=2
    )
    
    # B) Salience (CSR)
    fig.add_trace(
        go.Scatter(
            x=item_labels, 
            y=current['s_normalized'],
            fill='tozeroy',  # Filled area!
            mode='lines+markers',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10, color='#FF6B6B'),
            name="Salience (Competitive)"
        ),
        row=1, col=2
    )
    
    # 3. BOTTOM LEFT: DYNAMICS (Threshold)
    threshold_val = max(0, -current['nu'] / results['eta'])
    fig.add_trace(
        go.Bar(
            x=item_labels, y=y_effective, marker_color=colors, opacity=0.4, name="Claim"
        ),
        row=2, col=1
    )
    fig.add_hline(
        y=threshold_val, line_dash="dash", line_color="red", 
        annotation_text="Threshold", row=2, col=1
    )
    
    # 4. BOTTOM RIGHT: ACTIVE (Binary)
    fig.add_trace(
        go.Bar(
            x=item_labels, y=current['active'], 
            marker_color=[colors[i] if current['active'][i] else '#ccc' for i in range(5)],
            name="Active"
        ),
        row=2, col=2
    )
    
    # Layout Tweaks
    fig.update_layout(height=600, showlegend=True, 
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Response", row=1, col=1)
    fig.update_yaxes(title_text="Allocation Share", row=1, col=2, range=[0, 1.0])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    if break_equiv:
        st.info("💡 **Contrast:** Notice how the Gray Dotted Line (Simple Norm) never hits zero, but the Red Area (Salience) does. Salience forces a choice.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Indicators", f"{sum(current['active'])} / 5")
    col2.metric("Total Allocation", f"{current['s_sum']:.3f}", delta=f"{current['s_sum']-1:.3f}")
    
with tab2:
    st.markdown("## The Mathematics")
    st.latex(r"s_j^* = \max\left(0, \frac{d_j + \nu}{c}\right)")
    st.markdown(f"**Curvature (c):** {results['c']:.2f} | **Threshold (ν):** {final_nu:.2f}")

with tab3:
    # Water filling visual
    fig_deep = go.Figure()
    for i, (label, response) in enumerate(zip(item_labels, y_effective)):
        fig_deep.add_shape(type="rect", x0=i-0.3, x1=i+0.3, y0=0, y1=response, line=dict(color=colors[i], width=3))
        water_h = min(response, final_s[i] * eta)
        fig_deep.add_shape(type="rect", x0=i-0.28, x1=i+0.28, y0=0, y1=water_h, fillcolor=colors[i], opacity=0.5, line_width=0)
    
    t_h = max(0, -final_nu/eta)
    fig_deep.add_hline(y=t_h, line_dash="dash", line_color="blue", annotation_text="Water Level")
    fig_deep.update_layout(height=400, title="Water Level Visualization", showlegend=False)
    st.plotly_chart(fig_deep, use_container_width=True)

st.markdown("---")
st.markdown("**Reference:** *Competitive Salience–Reconstruction*")