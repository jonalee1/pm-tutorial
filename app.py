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

def water_filling_step_by_step(y, n_steps=50):
    """
    Compute salience via water-filling with step-by-step tracking.
    
    Returns history of threshold values and corresponding salience allocations.
    """
    J = len(y)
    eta = np.sum(np.maximum(y, 0.1))  # Volume estimate
    
    # Reconstruction drives: how much each indicator "wants" salience
    d = eta * y
    c = eta ** 2  # Curvature cost (same for all indicators here)
    
    # The unconstrained optimal for each indicator
    s_unconstrained = d / c  # = y / eta
    
    # Find the range for threshold search
    # Threshold ν must be such that s_j = max(0, (d_j + ν)/c) sums to 1
    nu_min = -np.max(d) - 1
    nu_max = c - np.min(d) + 1
    
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


def compute_final_salience(y):
    """Compute final salience via precise bisection."""
    J = len(y)
    eta = np.sum(np.maximum(y, 0.1))
    d = eta * y
    c = eta ** 2
    
    def salience_sum(nu):
        return np.sum(np.maximum(0, (d + nu) / c))
    
    # Bisection
    nu_low, nu_high = -np.max(d) - c, c
    for _ in range(100):
        nu_mid = (nu_low + nu_high) / 2
        if salience_sum(nu_mid) > 1:
            nu_high = nu_mid
        else:
            nu_low = nu_mid
    
    s = np.maximum(0, (d + nu_mid) / c)
    s = s / np.sum(s)
    return s, nu_mid, eta


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🌊 Competitive Salience-Reconstruction")
st.markdown("### How Indicators Compete for Measurement Attention")

st.markdown("""
This interactive demo shows the **water-filling algorithm** that determines 
how response emphasis is allocated across indicators in CSR.

**Key insight**: Indicators must "compete" for salience because the total must sum to 1.
Higher responses earn more salience, but the simplex constraint creates trade-offs.
""")

# Sidebar for input
st.sidebar.header("📊 Response Profile")
st.sidebar.markdown("Enter responses (1-5 scale) for each indicator:")

# Default example: heterogeneous profile
default_responses = [5, 4, 2, 1, 3]
item_labels = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]

responses = []
for i, (label, default) in enumerate(zip(item_labels, default_responses)):
    val = st.sidebar.slider(label, min_value=1, max_value=5, value=default, key=f"item_{i}")
    responses.append(val)

y = np.array(responses, dtype=float)

# Preset examples
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Examples:**")
col1, col2 = st.sidebar.columns(2)
if col1.button("Uniform"):
    st.session_state.update({f"item_{i}": 3 for i in range(5)})
    st.rerun()
if col2.button("Skewed"):
    for i, v in enumerate([5, 5, 2, 1, 1]):
        st.session_state[f"item_{i}"] = v
    st.rerun()

# Compute results
results = water_filling_step_by_step(y)
final_s, final_nu, eta = compute_final_salience(y)

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🎬 Animation", "📐 The Math", "🔬 Deep Dive"])

with tab1:
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
            "Response Profile (Input)",
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
        go.Bar(x=item_labels, y=y, marker_color=colors, name="Response"),
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
            y=y,
            marker_color=colors,
            name="Claim Height",
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add threshold line
    fig.add_hline(
        y=max(0, threshold_line), 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Threshold = {max(0, threshold_line):.2f}",
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
    st.markdown("### 🎯 What's Happening?")
    
    n_active = sum(current['active'])
    active_items = [item_labels[i] for i in range(5) if current['active'][i]]
    inactive_items = [item_labels[i] for i in range(5) if not current['active'][i]]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Indicators", f"{n_active} / 5")
    with col2:
        st.metric("Sum of Salience", f"{current['s_sum']:.3f}", 
                  delta=f"{current['s_sum'] - 1:.3f} from target")
    with col3:
        winner_idx = np.argmax(current['s_normalized'])
        st.metric("Current Winner", item_labels[winner_idx])
    
    if inactive_items:
        st.info(f"**Eliminated from competition:** {', '.join(inactive_items)} — their responses are below the threshold")
    
    if abs(current['s_sum'] - 1) < 0.05:
        st.success("✅ **Equilibrium reached!** The threshold is set so total salience = 1")


with tab2:
    st.markdown("## The Mathematics of Competition")
    
    st.markdown("""
    ### The Optimization Problem
    
    For a response vector $\mathbf{y}$ and volume $\eta$, we seek salience $\mathbf{s}$ that minimizes 
    reconstruction error subject to the simplex constraint:
    
    $$\min_{\mathbf{s}} \sum_{j=1}^{J} (y_j - \eta \cdot s_j)^2 \quad \text{subject to} \quad s_j \geq 0, \quad \sum_j s_j = 1$$
    
    ### The KKT Solution
    
    The closed-form solution via Karush-Kuhn-Tucker conditions is:
    
    $$s_j^* = \max\left(0, \\frac{d_j + \\nu}{c}\\right)$$
    
    where:
    - $d_j = \eta \cdot y_j$ is the **reconstruction drive** (how much indicator $j$ "wants" salience)
    - $c = \eta^2$ is the **curvature cost** (diminishing returns)
    - $\\nu$ is the **Lagrange multiplier** (the "water level" or threshold)
    
    ### The Competition Mechanism
    
    The threshold $\\nu$ is found by solving $\sum_j s_j^* = 1$. This creates competition:
    
    1. **High-response indicators** have high drives $d_j$, so they exceed the threshold easily
    2. **Low-response indicators** may fall below threshold and receive $s_j = 0$
    3. **The threshold adjusts** to ensure total allocation equals exactly 1
    """)
    
    # Show actual values
    st.markdown("### Your Current Values")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Reconstruction Drives ($d_j = \eta \cdot y_j$)**")
        for i, (label, d_val) in enumerate(zip(item_labels, results['d'])):
            st.write(f"{label}: {d_val:.2f}")
    
    with col2:
        st.markdown("**Final Salience Allocation**")
        for i, (label, s_val) in enumerate(zip(item_labels, final_s)):
            bar = "█" * int(s_val * 20)
            st.write(f"{label}: {s_val:.3f} ({s_val*100:.1f}%) {bar}")
    
    st.markdown(f"""
    **Volume (η):** {eta:.2f}  
    **Optimal Threshold (ν):** {final_nu:.4f}  
    **Curvature Cost (c):** {results['c']:.2f}
    """)


with tab3:
    st.markdown("## Deep Dive: The Water-Filling Metaphor")
    
    st.markdown("""
    Imagine each indicator as a **container** whose height is proportional to the response value.
    We pour "water" (salience) into these containers:
    
    1. Water rises uniformly across all containers
    2. Taller containers capture more water
    3. Short containers may get no water at all if the water level doesn't reach them
    4. The total water is fixed at 1 (the simplex constraint)
    """)
    
    # Create water-filling visualization
    fig = go.Figure()
    
    # Container outlines
    for i, (label, response) in enumerate(zip(item_labels, y)):
        # Container (rectangle)
        fig.add_shape(
            type="rect",
            x0=i-0.3, x1=i+0.3,
            y0=0, y1=response,
            line=dict(color=colors[i], width=3),
            fillcolor="rgba(255,255,255,0)"
        )
        
        # Water level (filled portion based on salience)
        water_height = final_s[i] * response / (final_s[i] if final_s[i] > 0 else 1)
        water_height = min(response, final_s[i] * eta)  # Actual reconstruction
        
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
        annotation_text="Water Level (threshold)"
    )
    
    fig.update_layout(
        title="Water-Filling Visualization",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(5)),
            ticktext=item_labels
        ),
        yaxis=dict(title="Response / Reconstruction"),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Key Insight: Why Competition Matters
    
    The simplex constraint ($\sum s_j = 1$) is what creates competition. Without it, 
    each indicator could receive unlimited salience. The constraint forces trade-offs:
    
    - **Zero-sum dynamics**: More salience to one indicator means less for others
    - **Threshold selection**: Only indicators "worthy" enough get positive allocation
    - **Automatic sparsity**: Low-response indicators may be eliminated entirely
    
    This competition is what makes salience meaningful as a measurement of **relative emphasis**.
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
**Reference:** This demo accompanies the paper *"Competitive Salience–Reconstruction: 
Separating Intensity and Shape in Psychological Measurement"*

The water-filling algorithm derives from convex optimization theory (Boyd & Vandenberghe, 2004) 
and has applications in information theory, resource allocation, and now—psychological measurement.
""")

## Requirements file (`requirements.txt`):
#streamlit>=1.28.0
#numpy>=1.24.0
#plotly>=5.18.0