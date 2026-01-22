"""
Competitive Reconstruction: Visualizing Water-Filling in CSR
=============================================================
Interactive demonstration of how indicators compete for salience allocation.

Deploy to Streamlit Cloud:
1. Create a GitHub repo with this file as `app.py`
2. Add a `requirements.txt` with: streamlit, numpy, matplotlib, plotly
3. Connect to share.streamlit.io
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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
    
    Args:
        y: Response vector
        n_steps: Number of steps for animation
        c_scale: Factor to scale curvature cost (1.0 = standard, <1.0 = sharper/aggressive)
    """
    J = len(y)
    eta = np.sum(np.maximum(y, 0.1))  # Volume estimate
    
    # Reconstruction drives: how much each indicator "wants" salience
    d = eta * y
    
    # Curvature cost: Standard is eta^2. Lowering this makes competition more aggressive.
    c = (eta ** 2) * c_scale
    
    # The unconstrained optimal for each indicator
    s_unconstrained = d / c 
    
    # Find the range for threshold search
    # Threshold ν must be such that s_j = max(0, (d_j + ν)/c) sums to 1
    # We broaden the search range to account for varying c
    nu_min = -np.max(d) - c
    nu_max = c - np.min(d) + c
    
    # Track history for visualization
    history = []
    
    # Bisection to find correct threshold
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
    
    # Find the step where s_sum is closest to 1
    sums = [h['s_sum'] for h in history]
    best_idx = np.argmin(np.abs(np.array(sums) - 1))
    
    return {
        'history': history,
        'best_idx': best_idx,
        'd': d,
        'c': c,
        'eta': eta,
        's_unconstrained': s_unconstrained,
        'y': y
    }


def compute_final_salience(y, c_scale=1.0):
    """Compute final salience via precise bisection."""
    J = len(y)
    eta = np.sum(np.maximum(y, 0.1))
    d = eta * y
    c = (eta ** 2) * c_scale
    
    def salience_sum(nu):
        return np.sum(np.maximum(0, (d + nu) / c))
    
    # Bisection
    nu_low, nu_high = -np.max(d) - c*2, c*2
    for _ in range(100):
        nu_mid = (nu_low + nu_high) / 2
        if salience_sum(nu_mid) > 1:
            nu_high = nu_mid
        else:
            nu_low = nu_mid
    
    s = np.maximum(0, (d + nu_mid) / c)
    # Ensure exact sum to 1 despite float errors
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

# Sidebar for input
st.sidebar.header("📊 Response Profile")
st.sidebar.markdown("Enter responses (1-5 scale):")

# -- Callbacks for Buttons --
def set_uniform():
    for i in range(5):
        st.session_state[f"item_{i}"] = 3

def set_skewed():
    vals = [5, 5, 2, 1, 1]
    for i, v in enumerate(vals):
        st.session_state[f"item_{i}"] = v

# Default example: heterogeneous profile
default_responses = [5, 4, 2, 1, 3]
item_labels = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]

responses = []
for i, (label, default) in enumerate(zip(item_labels, default_responses)):
    val = st.sidebar.slider(label, min_value=1, max_value=5, value=default, key=f"item_{i}")
    responses.append(val)

y_input = np.array(responses, dtype=float)

# Preset examples
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Examples:**")
col1, col2 = st.sidebar.columns(2)
col1.button("Uniform", on_click=set_uniform)
col2.button("Skewed", on_click=set_skewed)

# Break Equivalence Toggle
st.sidebar.markdown("---")
break_equiv = st.sidebar.checkbox(
    "Break Equivalence\n(Aggressive Competition)",
    value=False,
    help="Adds noise and increases competition to show how Salience differs from Normalization."
)

if break_equiv:
    st.sidebar.warning("⚠️ **Mode Active**: Adding noise and forcing sparsity.")
    # Add noise
    np.random.seed(42) # Fixed seed for consistency while toggling
    noise = np.random.normal(0, 0.4, size=y_input.shape)
    y_effective = np.maximum(0.1, y_input + noise)
    c_factor = 0.15 # Aggressive competition
else:
    y_effective = y_input
    c_factor = 1.0 # Standard

# Compute results
results = water_filling_step_by_step(y_effective, c_scale=c_factor)
final_s, final_nu, eta = compute_final_salience(y_effective, c_scale=c_factor)
simple_norm = y_effective / np.sum(y_effective)

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

tab_contrast, tab_anim, tab_math, tab_deep = st.tabs(["⚖️ Contrast", "🎬 Animation", "📐 The Math", "🔬 Deep Dive"])

with tab_contrast:
    st.markdown("### Simple Normalization vs. Competitive Salience")
    st.markdown("""
    Compare **Simple Normalization** (just dividing by the sum) with **Salience** (water-filling).
    
    * **Standard Mode**: They look similar. This is why people confuse them.
    * **Break Equivalence Mode**: Toggle the sidebar option to see them diverge. Normalization stays smooth; Salience forces choices.
    """)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Raw Responses (Ghosted/Background) - Secondary Axis
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=y_effective,
            name="Raw Response (1-5)",
            marker_color='gray',
            opacity=0.2,
            offsetgroup=0
        ),
        secondary_y=True
    )
    
    # 2. Simple Normalization
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=simple_norm,
            name="Simple Normalization",
            marker_color='#45B7D1',  # Blue
            offsetgroup=1
        ),
        secondary_y=False
    )
    
    # 3. Salience (CSR)
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=final_s,
            name="Salience (Water-Filling)",
            marker_color='#FF6B6B',  # Red
            offsetgroup=2
        ),
        secondary_y=False
    )
    
    fig.update_layout(
        barmode='group',
        height=500,
        title_text="Allocating Attention: Proportional vs. Competitive",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Allocation (Probability)", secondary_y=False, range=[0, max(1.0, max(final_s)*1.1)])
    fig.update_yaxes(title_text="Raw Scale (1-5)", secondary_y=True, showgrid=False, range=[0, 6])

    st.plotly_chart(fig, use_container_width=True)
    
    if break_equiv:
        st.info("""
        **Observation:** Notice how **Simple Normalization** keeps every item alive, merely scaling the noise. 
        In contrast, **Salience** eliminates the weak signals entirely (forcing sparsity) and amplifies the winners.
        This "winner-take-all" dynamic is the essence of competition.
        """)


with tab_anim:
    st.markdown("## Watch Indicators Compete")
    
    # Animation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        step = st.slider(
            "Water Level (Threshold ν)", 
            min_value=0, 
            max_value=len(results['history'])-1, 
            value=results['best_idx'],
            help="Drag to see how the threshold determines allocation"
        )
    
    current = results['history'][step]
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Response Profile (Effective Input)",
            "Salience Allocation (Output)",
            "Competition Dynamics",
            "Who's Active?"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Panel 1: Response profile
    fig.add_trace(
        go.Bar(x=item_labels, y=y_effective, marker_color=colors, name="Response"),
        row=1, col=1
    )
    
    # Panel 2: Current salience allocation
    fig.add_trace(
        go.Bar(
            x=item_labels, 
            y=current['s_normalized'] * 100,
            marker_color=[colors[i] if current['active'][i] else '#CCCCCC' for i in range(5)],
            name="Salience %",
            text=[f"{v:.1f}%" for v in current['s_normalized'] * 100],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # Panel 3: Competition dynamics - show drives vs threshold
    threshold_line = -current['nu'] / results['eta']  # Convert back to response scale
    
    fig.add_trace(
        go.Bar(
            x=item_labels, 
            y=y_effective,
            marker_color=colors,
            name="Claim Height",
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add threshold line
    threshold_h_val = max(0, threshold_line)
    fig.add_hline(
        y=threshold_h_val, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Threshold",
        row=2, col=1
    )
    
    # Panel 4: Active indicators
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=current['active'],
            marker_color=[colors[i] if current['active'][i] else '#CCCCCC' for i in range(5)],
            name="Active"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text=f"Step {step}: Sum of raw salience = {current['s_sum']:.3f} (target: 1.000)"
    )
    
    fig.update_yaxes(title_text="Response", row=1, col=1)
    fig.update_yaxes(title_text="Salience (%)", row=1, col=2, range=[0, 100])
    fig.update_yaxes(title_text="Response", row=2, col=1)
    fig.update_yaxes(title_text="Active (1/0)", row=2, col=2, range=[0, 1.5])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    n_active = sum(current['active'])
    active_items = [item_labels[i] for i in range(5) if current['active'][i]]
    inactive_items = [item_labels[i] for i in range(5) if not current['active'][i]]
    
    if inactive_items:
        st.info(f"**Eliminated:** {', '.join(inactive_items)}")
    
    if abs(current['s_sum'] - 1) < 0.05:
        st.success("✅ **Equilibrium reached!**")


with tab_math:
    st.markdown("## The Mathematics of Competition")
    
    st.markdown(r"""
    ### The Optimization Problem
    
    Minimizing reconstruction error subject to the simplex constraint yields the KKT solution:
    
    $$s_j^* = \max\left(0, \frac{d_j + \nu}{c}\right)$$
    
    where $d_j = \eta \cdot y_j$ is the **drive** and $c = \eta^2$ is the **cost**.
    
    ### Breaking Equivalence
    
    When we "Break Equivalence", we lower the cost $c$ (or increase the drives relative to cost). 
    This makes the slope of the water-filling function steeper.
    
    - **High Cost (Standard)**: The function is flat. Small differences in $y$ create small differences in $s$. Result $\approx$ Proportional.
    - **Low Cost (Aggressive)**: The function is steep. Small differences in $y$ push some items below the threshold. Result $\neq$ Proportional.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Curvature Cost (c)", f"{results['c']:.2f}")
    with col2:
        st.metric("Threshold (ν)", f"{final_nu:.2f}")


with tab_deep:
    st.markdown("## Deep Dive: Water-Filling Visualization")
    
    fig = go.Figure()
    
    # Container outlines
    for i, (label, response) in enumerate(zip(item_labels, y_effective)):
        fig.add_shape(
            type="rect",
            x0=i-0.3, x1=i+0.3,
            y0=0, y1=response,
            line=dict(color=colors[i], width=3),
            fillcolor="rgba(255,255,255,0)"
        )
        
        # Water level
        water_height = min(response, final_s[i] * eta)
        
        fig.add_shape(
            type="rect",
            x0=i-0.28, x1=i+0.28,
            y0=0, y1=final_s[i] * eta,
            fillcolor=colors[i],
            opacity=0.5,
            line=dict(width=0)
        )
    
    # Threshold line
    threshold_height = max(0, -final_nu / eta)
    fig.add_hline(
        y=threshold_height,
        line_dash="dash",
        line_color="blue",
        annotation_text="Water Level"
    )
    
    fig.update_layout(
        title="Water-Filling Visualization",
        xaxis=dict(tickmode='array', tickvals=list(range(5)), ticktext=item_labels),
        yaxis=dict(title="Response / Reconstruction"),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("The dashed line represents the water level. Any container shorter than this line gets no water (salience = 0).")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
**Reference:** This demo accompanies the paper *"Competitive Salience–Reconstruction: 
Separating Intensity and Shape in Psychological Measurement"*
""")