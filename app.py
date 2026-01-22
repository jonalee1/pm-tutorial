"""
Competitive Salience-Reconstruction: Full Estimation Demo
=========================================================
Interactive demonstration of CSR's alternating optimization algorithm.

The key insight: Volume and Salience mutually determine each other.
This app shows the iterative process that finds their self-consistent values.

Deploy to Streamlit Cloud:
1. Create a GitHub repo with this file as `app.py`
2. Add `requirements.txt` with: streamlit, numpy, plotly
3. Connect to share.streamlit.io
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="CSR Estimation Demo",
    page_icon="🔄",
    layout="wide"
)

# ============================================================================
# CORE CSR ALGORITHM — FULL ALTERNATING OPTIMIZATION
# ============================================================================

def initialize_salience(y, method='linear'):
    """Initialize salience for alternating optimization."""
    J = len(y)
    
    if method == 'uniform':
        return np.ones(J) / J
    
    elif method == 'linear':
        # Linear competitive: above-average responses get salience
        y_centered = y - np.mean(y)
        s_raw = np.maximum(0, y_centered)
        if np.sum(s_raw) > 0:
            return s_raw / np.sum(s_raw)
        else:
            return np.ones(J) / J
    
    elif method == 'proportional':
        # Simple normalization (what we're contrasting against)
        y_pos = np.maximum(y, 0)
        return y_pos / np.sum(y_pos)
    
    return np.ones(J) / J


def update_volume(y, s):
    """
    Update volume given current salience.
    
    Closed-form solution from minimizing reconstruction error:
    η = (Σ y_j s_j) / (Σ s_j²)
    """
    numerator = np.sum(y * s)
    denominator = np.sum(s ** 2)
    
    if denominator < 1e-10:
        return np.sum(y)  # Fallback to sum score
    
    return numerator / denominator


def update_salience_waterfill(y, eta, return_details=False):
    """
    Update salience given current volume via water-filling.
    
    Solves: min_s Σ(y_j - η·s_j)² subject to s ∈ Δ^{J-1}
    
    KKT solution: s_j* = max(0, (d_j + ν) / c)
    where d_j = η·y_j (reconstruction drive), c = η² (curvature)
    """
    J = len(y)
    
    if eta < 1e-10:
        s = np.ones(J) / J
        if return_details:
            return s, {'threshold': 0, 'active': np.ones(J), 'drives': y}
        return s
    
    # Reconstruction drives
    d = eta * y
    c = eta ** 2
    
    # Bisection to find threshold ν such that Σ s_j = 1
    nu_low = -np.max(d) - c
    nu_high = c - np.min(d) + c
    
    for _ in range(100):
        nu_mid = (nu_low + nu_high) / 2
        s_raw = np.maximum(0, (d + nu_mid) / c)
        s_sum = np.sum(s_raw)
        
        if s_sum > 1:
            nu_high = nu_mid
        else:
            nu_low = nu_mid
        
        if abs(s_sum - 1) < 1e-10:
            break
    
    # Final salience
    s = np.maximum(0, (d + nu_mid) / c)
    if np.sum(s) > 0:
        s = s / np.sum(s)  # Ensure exactly sums to 1
    else:
        s = np.ones(J) / J
    
    if return_details:
        # Convert threshold back to response scale for visualization
        threshold_response_scale = -nu_mid / eta if eta > 0 else 0
        details = {
            'threshold': threshold_response_scale,
            'threshold_nu': nu_mid,
            'active': (s > 1e-10).astype(int),
            'drives': d,
            'curvature': c
        }
        return s, details
    
    return s


def csr_estimate(y, max_iter=50, tol=1e-8, init_method='linear'):
    """
    Full CSR estimation via alternating optimization.
    
    Returns complete iteration history for visualization.
    """
    J = len(y)
    
    # Initialize
    s = initialize_salience(y, method=init_method)
    eta = update_volume(y, s)
    
    history = []
    
    for iteration in range(max_iter):
        # Store current state
        s_old = s.copy()
        eta_old = eta
        
        # Reconstruction with current estimates
        y_hat = eta * s
        recon_error = np.sum((y - y_hat) ** 2)
        
        # Get salience details for visualization
        _, salience_details = update_salience_waterfill(y, eta, return_details=True)
        
        history.append({
            'iteration': iteration,
            'eta': eta,
            's': s.copy(),
            'y_hat': y_hat.copy(),
            'recon_error': recon_error,
            'threshold': salience_details['threshold'],
            'active': salience_details['active'].copy(),
            'delta_eta': abs(eta - eta_old) if iteration > 0 else np.nan,
            'delta_s': np.max(np.abs(s - s_old)) if iteration > 0 else np.nan
        })
        
        # Check convergence
        if iteration > 0:
            if abs(eta - eta_old) < tol and np.max(np.abs(s - s_old)) < tol:
                break
        
        # Update volume given salience
        eta = update_volume(y, s)
        
        # Update salience given volume
        s = update_salience_waterfill(y, eta)
    
    return {
        'history': history,
        'final_eta': eta,
        'final_s': s,
        'converged': len(history) < max_iter,
        'n_iterations': len(history)
    }


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🔄 CSR: Self-Consistent Estimation")
st.markdown("### Watch Volume and Salience Find Their Equilibrium")

st.markdown("""
**Competitive Salience-Reconstruction** decomposes responses into:
- **Volume (η)**: Overall response intensity ("how much")
- **Salience (s)**: Relative allocation across indicators ("where")

These two components **mutually determine each other**. This app shows the 
alternating optimization that finds their self-consistent values.
""")

# ============================================================================
# SIDEBAR — INPUT CONTROLS
# ============================================================================

st.sidebar.header("📊 Response Profile")

# Preset examples
st.sidebar.markdown("**Quick Presets:**")
preset = st.sidebar.selectbox(
    "Select a pattern",
    ["Custom", "Uniform (3,3,3,3,3)", "Skewed (5,5,2,1,1)", 
     "Bimodal (5,1,1,1,5)", "Gradient (5,4,3,2,1)"],
    index=0
)

presets = {
    "Uniform (3,3,3,3,3)": [3, 3, 3, 3, 3],
    "Skewed (5,5,2,1,1)": [5, 5, 2, 1, 1],
    "Bimodal (5,1,1,1,5)": [5, 1, 1, 1, 5],
    "Gradient (5,4,3,2,1)": [5, 4, 3, 2, 1],
}

if preset != "Custom":
    default_responses = presets[preset]
else:
    default_responses = [5, 4, 2, 1, 3]

st.sidebar.markdown("**Adjust responses:**")
item_labels = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]

responses = []
for i, (label, default) in enumerate(zip(item_labels, default_responses)):
    val = st.sidebar.slider(
        label, 
        min_value=1, 
        max_value=5, 
        value=default, 
        key=f"response_{i}_{preset}"
    )
    responses.append(val)

y = np.array(responses, dtype=float)

# Algorithm settings
st.sidebar.markdown("---")
st.sidebar.markdown("**Algorithm Settings:**")
init_method = st.sidebar.selectbox(
    "Initialization",
    ["linear", "uniform", "proportional"],
    help="How to initialize salience before optimization"
)

# ============================================================================
# RUN ESTIMATION
# ============================================================================

results = csr_estimate(y, init_method=init_method)
history = results['history']

# ============================================================================
# MAIN DISPLAY
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Iteration View", 
    "📈 Convergence", 
    "⚔️ Competition in Action",
    "📐 The Algorithm"
])

# ----------------------------------------------------------------------------
# TAB 1: ITERATION VIEW
# ----------------------------------------------------------------------------
with tab1:
    st.markdown("## Step Through the Estimation")
    
    n_iter = len(history)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        iteration = st.slider(
            "Iteration", 
            min_value=0, 
            max_value=n_iter - 1, 
            value=min(n_iter - 1, 5),
            help="Step through the alternating optimization"
        )
    with col2:
        st.metric("Total Iterations", n_iter)
    
    current = history[iteration]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Create 2x2 visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Input Responses",
            f"Salience Allocation (Iteration {iteration})",
            f"Reconstruction: y ≈ η·s",
            f"Active Indicators (above threshold)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Panel 1: Input responses
    fig.add_trace(
        go.Bar(
            x=item_labels, 
            y=y, 
            marker_color=colors,
            text=[f"{v:.0f}" for v in y],
            textposition='outside',
            name="Response"
        ),
        row=1, col=1
    )
    
    # Panel 2: Salience
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=current['s'] * 100,
            marker_color=[colors[i] if current['active'][i] else '#CCCCCC' for i in range(5)],
            text=[f"{v:.1f}%" for v in current['s'] * 100],
            textposition='outside',
            name="Salience"
        ),
        row=1, col=2
    )
    
    # Panel 3: Reconstruction
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=y,
            marker_color=colors,
            opacity=0.3,
            name="Observed (y)"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=current['y_hat'],
            marker_color=colors,
            opacity=0.8,
            name="Reconstructed (η·s)"
        ),
        row=2, col=1
    )
    
    # Panel 4: Competition - show threshold cutting off items
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=y,
            marker_color=[colors[i] if current['active'][i] else '#CCCCCC' for i in range(5)],
            name="Response"
        ),
        row=2, col=2
    )
    
    # Add threshold line
    if current['threshold'] > 0:
        fig.add_hline(
            y=current['threshold'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold = {current['threshold']:.2f}",
            row=2, col=2
        )
    
    fig.update_layout(
        height=550,
        showlegend=False,
        title_text=f"Iteration {iteration}: η = {current['eta']:.3f}, Error = {current['recon_error']:.4f}"
    )
    
    fig.update_yaxes(title_text="Response", row=1, col=1, range=[0, 6])
    fig.update_yaxes(title_text="Salience (%)", row=1, col=2, range=[0, 60])
    fig.update_yaxes(title_text="Value", row=2, col=1, range=[0, 6])
    fig.update_yaxes(title_text="Response", row=2, col=2, range=[0, 6])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics row
    st.markdown("### 📊 Current State")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Volume (η)", 
            f"{current['eta']:.3f}",
            delta=f"{current['delta_eta']:.4f}" if iteration > 0 else None
        )
    with col2:
        st.metric("Sum Score", f"{np.sum(y):.0f}")
    with col3:
        n_active = int(np.sum(current['active']))
        st.metric("Active Items", f"{n_active} / 5")
    with col4:
        st.metric(
            "Recon. Error", 
            f"{current['recon_error']:.4f}"
        )
    
    # Show the two update equations
    st.markdown("---")
    st.markdown("### 🔄 The Two Update Steps")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Volume Update** (given salience):
        
        $$\\eta = \\frac{\\sum_j y_j \\cdot s_j}{\\sum_j s_j^2}$$
        
        *"What intensity best explains responses given this allocation?"*
        """)
    
    with col2:
        st.markdown("""
        **Salience Update** (given volume):
        
        $$s_j^* = \\max\\left(0, \\frac{\\eta \\cdot y_j + \\nu}{\\eta^2}\\right)$$
        
        *"What allocation best reconstructs responses given this intensity?"*
        """)


# ----------------------------------------------------------------------------
# TAB 2: CONVERGENCE
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("## Convergence Dynamics")
    
    st.markdown("""
    Watch how volume and salience converge to their self-consistent values.
    The algorithm finds a **fixed point** where each component is optimal given the other.
    """)
    
    iterations = [h['iteration'] for h in history]
    etas = [h['eta'] for h in history]
    errors = [h['recon_error'] for h in history]
    
    # Salience trajectories
    salience_traces = {f"s_{j+1}": [h['s'][j] for h in history] for j in range(5)}
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Volume (η) Convergence",
            "Reconstruction Error",
            "Salience Trajectories",
            "Final Decomposition"
        ),
        vertical_spacing=0.15
    )
    
    # Volume convergence
    fig.add_trace(
        go.Scatter(
            x=iterations, y=etas,
            mode='lines+markers',
            name='Volume',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    fig.add_hline(
        y=np.sum(y), 
        line_dash="dot", 
        line_color="gray",
        annotation_text="Sum Score",
        row=1, col=1
    )
    
    # Reconstruction error
    fig.add_trace(
        go.Scatter(
            x=iterations, y=errors,
            mode='lines+markers',
            name='Error',
            line=dict(color='#3498DB', width=3),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Salience trajectories
    for j, (name, vals) in enumerate(salience_traces.items()):
        fig.add_trace(
            go.Scatter(
                x=iterations, y=vals,
                mode='lines+markers',
                name=item_labels[j],
                line=dict(color=colors[j], width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
    
    # Final decomposition bar chart
    final = history[-1]
    fig.add_trace(
        go.Bar(
            x=item_labels,
            y=final['s'] * 100,
            marker_color=colors,
            name='Final Salience',
            text=[f"{v:.1f}%" for v in final['s'] * 100],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_yaxes(title_text="η", row=1, col=1)
    fig.update_yaxes(title_text="MSE", row=1, col=2)
    fig.update_yaxes(title_text="Salience", row=2, col=1)
    fig.update_yaxes(title_text="Salience (%)", row=2, col=2)
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Convergence summary
    st.markdown("### Convergence Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iterations to Converge", results['n_iterations'])
    with col2:
        st.metric("Final Volume", f"{results['final_eta']:.4f}")
    with col3:
        st.metric("Final Error", f"{history[-1]['recon_error']:.6f}")
    
    # Compare to sum score
    st.markdown("---")
    st.markdown("### Volume vs. Sum Score")
    
    sum_score = np.sum(y)
    final_eta = results['final_eta']
    
    st.markdown(f"""
    | Measure | Value |
    |---------|-------|
    | Sum Score | {sum_score:.2f} |
    | CSR Volume (η) | {final_eta:.4f} |
    | Difference | {final_eta - sum_score:.4f} |
    
    **Note:** Under uniform salience, volume equals the sum score exactly.
    The difference reflects how non-uniform salience reallocates the intensity measure.
    """)


# ----------------------------------------------------------------------------
# TAB 3: COMPETITION IN ACTION
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("## ⚔️ Competition in Action")
    
    st.markdown("""
    **Key insight:** Salience is NOT just normalization. 
    
    The simplex constraint creates genuine competition: indicators must "earn" their 
    allocation by contributing to reconstruction. Low contributors get **eliminated**.
    """)
    
    st.info("💡 **Try this:** Set one item very low (e.g., Item 4 = 1) while others are high. Watch it get zero salience!")
    
    final = history[-1]
    
    # Side-by-side comparison: Normalization vs CSR
    col1, col2 = st.columns(2)
    
    # Proportional normalization (what people think salience is)
    y_pos = np.maximum(y, 0)
    normalized = y_pos / np.sum(y_pos)
    
    with col1:
        st.markdown("### Simple Normalization")
        st.markdown("*Just divide by sum — every item gets something*")
        
        fig_norm = go.Figure()
        fig_norm.add_trace(go.Bar(
            x=item_labels,
            y=normalized * 100,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in normalized * 100],
            textposition='outside'
        ))
        fig_norm.update_layout(
            height=300,
            yaxis_title="Normalized (%)",
            yaxis_range=[0, 50]
        )
        st.plotly_chart(fig_norm, use_container_width=True)
        
        st.markdown(f"**All items active:** 5 / 5")
        st.markdown(f"**Min allocation:** {np.min(normalized)*100:.1f}%")
    
    with col2:
        st.markdown("### CSR Salience")
        st.markdown("*Competitive reconstruction — losers get zero*")
        
        fig_csr = go.Figure()
        fig_csr.add_trace(go.Bar(
            x=item_labels,
            y=final['s'] * 100,
            marker_color=[colors[i] if final['active'][i] else '#CCCCCC' for i in range(5)],
            text=[f"{v:.1f}%" for v in final['s'] * 100],
            textposition='outside'
        ))
        fig_csr.update_layout(
            height=300,
            yaxis_title="Salience (%)",
            yaxis_range=[0, 50]
        )
        st.plotly_chart(fig_csr, use_container_width=True)
        
        n_active = int(np.sum(final['active']))
        st.markdown(f"**Active items:** {n_active} / 5")
        st.markdown(f"**Items eliminated:** {5 - n_active}")
    
    # The threshold mechanism
    st.markdown("---")
    st.markdown("### 🎯 The Threshold Mechanism")
    
    st.markdown("""
    The water-filling algorithm sets a **threshold** such that:
    - Items above threshold receive salience proportional to their excess
    - Items below threshold receive **exactly zero**
    - Total salience sums to 1 (simplex constraint)
    
    This is what creates competition: the threshold adjusts based on ALL items.
    """)
    
    fig_threshold = go.Figure()
    
    # Bars for responses
    fig_threshold.add_trace(go.Bar(
        x=item_labels,
        y=y,
        marker_color=[colors[i] if final['active'][i] else '#CCCCCC' for i in range(5)],
        name='Response'
    ))
    
    # Threshold line
    threshold = final['threshold']
    if threshold > 0:
        fig_threshold.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Threshold = {threshold:.2f}"
        )
        
        # Annotate what's above/below
        for i, (label, resp) in enumerate(zip(item_labels, y)):
            if resp > threshold:
                excess = resp - threshold
                fig_threshold.add_annotation(
                    x=label, y=resp + 0.3,
                    text=f"+{excess:.1f}",
                    showarrow=False,
                    font=dict(color='green', size=12)
                )
    
    fig_threshold.update_layout(
        height=350,
        title="Items Compete Against the Threshold",
        yaxis_title="Response",
        yaxis_range=[0, 6],
        showlegend=False
    )
    
    st.plotly_chart(fig_threshold, use_container_width=True)
    
    # Explain the difference
    st.markdown("---")
    st.markdown("### 🔑 Why This Matters")
    
    diff = np.abs(final['s'] - normalized)
    max_diff_idx = np.argmax(diff)
    
    st.markdown(f"""
    **Maximum difference between normalization and CSR:** {np.max(diff)*100:.1f}% 
    (at {item_labels[max_diff_idx]})
    
    This difference reveals what CSR captures that normalization misses:
    - **Normalization** treats all responses as contributing proportionally
    - **CSR** identifies which items are *actually diagnostic* given the response pattern
    
    When responses are relatively uniform, the methods converge.  
    When responses are differentiated, CSR reveals the competitive structure.
    """)


# ----------------------------------------------------------------------------
# TAB 4: THE ALGORITHM
# ----------------------------------------------------------------------------
with tab4:
    st.markdown("## 📐 The CSR Algorithm")
    
    st.markdown("""
    ### The Self-Consistency Principle
    
    CSR seeks a decomposition $(\\eta, \\mathbf{s})$ where each component is optimal given the other.
    This is a **fixed-point** condition, not parameter recovery.
    """)
    
    st.markdown("""
    ### Algorithm: Alternating Optimization
```
    1. Initialize salience s (e.g., linear competitive)
    
    2. Repeat until convergence:
       
       a. UPDATE VOLUME given salience:
          η = (Σⱼ yⱼ·sⱼ) / (Σⱼ sⱼ²)
          
       b. UPDATE SALIENCE given volume (water-filling):
          For each item j:
            sⱼ = max(0, (η·yⱼ + ν) / η²)
          where ν is set so Σⱼ sⱼ = 1
          
    3. Return (η, s) at convergence
```
    """)
    
    st.markdown("---")
    st.markdown("### Why Alternating Optimization Works")
    
    st.markdown("""
    Each update **decreases reconstruction error** while respecting constraints:
    
    | Step | Optimization Problem | Constraint |
    |------|---------------------|------------|
    | Volume update | min_η Σ(yⱼ - η·sⱼ)² | η > 0 |
    | Salience update | min_s Σ(yⱼ - η·sⱼ)² | s ∈ Δ^{J-1} |
    
    **Convergence guarantee:** The objective decreases monotonically, and the 
    problem is convex in each variable separately (though not jointly).
    """)
    
    st.markdown("---")
    st.markdown("### The Water-Filling Solution")
    
    st.markdown("""
    The salience update has a **closed-form solution** via KKT conditions:
    
    $$s_j^* = \\max\\left(0, \\frac{d_j + \\nu}{c}\\right)$$
    
    where:
    - $d_j = \\eta \\cdot y_j$ — **Reconstruction drive** (incentive to allocate to item j)
    - $c = \\eta^2$ — **Curvature cost** (diminishing returns)
    - $\\nu$ — **Lagrange multiplier** (the "water level" threshold)
    
    The threshold $\\nu$ is found by bisection to satisfy $\\sum_j s_j = 1$.
    """)
    
    st.markdown("---")
    st.markdown("### Your Current Estimation")
    
    final = history[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Reconstruction Drives** $(d_j = \\eta \\cdot y_j)$")
        eta = final['eta']
        for i, (label, resp) in enumerate(zip(item_labels, y)):
            drive = eta * resp
            st.write(f"{label}: {drive:.3f}")
    
    with col2:
        st.markdown("**Final Salience Allocation**")
        for i, (label, s_val) in enumerate(zip(item_labels, final['s'])):
            bar = "█" * int(s_val * 30)
            status = "✓" if final['active'][i] else "✗"
            st.write(f"{status} {label}: {s_val:.3f} ({s_val*100:.1f}%) {bar}")
    
    st.markdown(f"""
    ---
    **Final Parameters:**
    - Volume (η): {final['eta']:.4f}
    - Threshold (in response scale): {final['threshold']:.3f}
    - Reconstruction Error: {final['recon_error']:.6f}
    - Converged in: {results['n_iterations']} iterations
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
**Reference:** This demo accompanies *"Competitive Salience–Reconstruction: 
Separating Intensity and Shape in Psychological Measurement"*

**Key takeaway:** CSR estimation finds the self-consistent decomposition where 
volume and salience mutually determine each other. This is fundamentally different 
from simple normalization — it's optimization under constraint.
""")