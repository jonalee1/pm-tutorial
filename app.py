import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="CSR: Competitive Salience", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stAlert {background-color: #1a1a2e; border: 1px solid #4a4a6a;}
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #e94560;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .key-message {
        background: #0f3460;
        border: 2px solid #e94560;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-size: 1.1em;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

def water_filling_salience(responses, threshold_v):
    """
    Compute salience via water-filling algorithm.
    Returns salience vector and active indicator mask.
    """
    J = len(responses)
    responses = np.array(responses, dtype=float)
    
    # Shift responses by threshold (water level)
    shifted = responses - threshold_v
    
    # Only positive contributions are "above water"
    above_water = np.maximum(0, shifted)
    
    # Normalize to simplex (if any are above water)
    total = above_water.sum()
    if total > 0:
        salience = above_water / total
    else:
        # All underwater - uniform fallback
        salience = np.ones(J) / J
    
    active = above_water > 0
    return salience, active, above_water

def simple_normalization(responses):
    """Simple proportional normalization (what people confuse with salience)"""
    responses = np.array(responses, dtype=float)
    return responses / responses.sum()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("📊 Response Profile")
    st.caption("Enter responses (1-5 scale) for each indicator")
    
    # Quick presets
    st.subheader("Quick Examples")
    col1, col2, col3 = st.columns(3)
    
    if col1.button("Uniform", use_container_width=True):
        st.session_state.responses = [3, 3, 3, 3, 3]
    if col2.button("Skewed", use_container_width=True):
        st.session_state.responses = [5, 4, 2, 1, 1]
    if col3.button("Extreme", use_container_width=True):
        st.session_state.responses = [5, 5, 1, 1, 1]
    
    # Initialize responses
    if 'responses' not in st.session_state:
        st.session_state.responses = [5, 4, 3, 2, 1]
    
    st.divider()
    
    # Response sliders
    responses = []
    for i in range(5):
        val = st.slider(
            f"Item {i+1}", 
            min_value=1, 
            max_value=5, 
            value=st.session_state.responses[i],
            key=f"item_{i}"
        )
        responses.append(val)
    st.session_state.responses = responses
    
    st.divider()
    
    # Stress test controls
    st.subheader("⚡ Stress Test")
    st.caption("Break the 'normalization equivalence'")
    
    threshold_v = st.slider(
        "Water Level (threshold)", 
        min_value=0.0, 
        max_value=4.0, 
        value=1.0,
        step=0.1,
        help="Raise this to see indicators drop out"
    )
    
    add_noise = st.checkbox("Add measurement noise", value=False)
    noise_level = 0.0
    if add_noise:
        noise_level = st.slider("Noise SD", 0.1, 1.0, 0.3)

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("Competitive Salience–Reconstruction")

# Key message box
st.markdown("""
<div class="key-message">
    💡 <strong>If this looks like simple normalization, raise the water level.</strong><br>
    <small>Competition becomes visible when indicators start dropping out.</small>
</div>
""", unsafe_allow_html=True)

# Apply noise if requested
working_responses = np.array(responses, dtype=float)
if add_noise and noise_level > 0:
    np.random.seed(42)  # For reproducibility in demo
    working_responses = working_responses + np.random.normal(0, noise_level, 5)
    working_responses = np.clip(working_responses, 0.5, 5.5)

# Compute both methods
salience, active, above_water = water_filling_salience(working_responses, threshold_v)
normalized = simple_normalization(working_responses)

# Count active indicators
n_active = active.sum()

# =============================================================================
# VISUALIZATION
# =============================================================================

tab1, tab2, tab3 = st.tabs(["🎯 The Competition", "📐 The Math", "🔬 Deep Comparison"])

with tab1:
    st.subheader("Watch Indicators Compete for Salience")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### Panel A: Input Responses")
        st.caption("What you observed")
        
        # Input bar chart with water level line
        fig_input = go.Figure()
        
        colors_input = ['#4ecdc4' if a else '#95a5a6' for a in active]
        
        fig_input.add_trace(go.Bar(
            x=[f"Item {i+1}" for i in range(5)],
            y=working_responses,
            marker_color=colors_input,
            name="Response",
            text=[f"{v:.1f}" for v in working_responses],
            textposition='outside'
        ))
        
        # Add water level line
        fig_input.add_hline(
            y=threshold_v, 
            line_dash="dash", 
            line_color="#e94560",
            line_width=3,
            annotation_text=f"Water Level = {threshold_v:.1f}",
            annotation_position="top right",
            annotation_font_color="#e94560"
        )
        
        # Shade the "underwater" region
        fig_input.add_hrect(
            y0=0, y1=threshold_v,
            fillcolor="#e94560", opacity=0.15,
            line_width=0,
            annotation_text="Below water = inactive",
            annotation_position="bottom left"
        )
        
        fig_input.update_layout(
            height=350,
            yaxis_range=[0, 6],
            yaxis_title="Response Value",
            showlegend=False,
            template="plotly_dark",
            margin=dict(t=30, b=30)
        )
        st.plotly_chart(fig_input, use_container_width=True)
    
    with col_right:
        st.markdown("### Panel B: Competitive Salience")
        st.caption("How reconstruction allocates capacity")
        
        # Salience bar chart
        fig_salience = go.Figure()
        
        # Color by active/inactive
        colors_salience = ['#00d4aa' if a else '#dc3545' for a in active]
        
        fig_salience.add_trace(go.Bar(
            x=[f"Item {i+1}" for i in range(5)],
            y=salience * 100,
            marker_color=colors_salience,
            name="Salience",
            text=[f"{s*100:.1f}%" if a else "OUT" for s, a in zip(salience, active)],
            textposition='outside',
            textfont=dict(size=14, color=['white' if a else '#dc3545' for a in active])
        ))
        
        fig_salience.update_layout(
            height=350,
            yaxis_range=[0, 100],
            yaxis_title="Salience (%)",
            showlegend=False,
            template="plotly_dark",
            margin=dict(t=30, b=30)
        )
        st.plotly_chart(fig_salience, use_container_width=True)
    
    # Status indicator
    if n_active < 5:
        st.success(f"🎯 **Competition is visible!** {5 - n_active} indicator(s) eliminated. Salience ≠ Normalization.")
    else:
        st.warning("⚠️ All indicators still active. Raise the water level to see competition eliminate weak indicators.")
    
    # The key insight box
    st.markdown("""
    <div class="insight-box">
        <strong>🔑 Why this isn't normalization:</strong><br>
        Simple normalization would give Item 5 some weight even if it's barely endorsed.<br>
        Competitive salience can drive indicators to <strong>exactly zero</strong> — 
        they're eliminated from measurement because they don't clear the threshold.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.subheader("The Water-Filling Mechanism")
    
    st.markdown("""
    **The Algorithm:**
    1. Set a "water level" threshold $\\nu$
    2. Each indicator's contribution = $\\max(0, y_j - \\nu)$
    3. Indicators below water get **exactly zero** salience
    4. Remaining capacity is distributed among active indicators
    
    **The Key Difference:**
    - **Normalization**: $s_j = y_j / \\sum_k y_k$ — everyone gets something
    - **Competitive Salience**: $s_j = \\max(0, y_j - \\nu) / \\sum_k \\max(0, y_k - \\nu)$ — weak indicators get nothing
    """)
    
    # Show the math visually
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Responses above water:**")
        for i, (y, aw, a) in enumerate(zip(working_responses, above_water, active)):
            status = "✅" if a else "❌"
            st.write(f"Item {i+1}: {y:.2f} - {threshold_v:.1f} = {aw:.2f} {status}")
    
    with col2:
        st.markdown("**Salience allocation:**")
        st.write(f"Sum of above-water: {above_water.sum():.2f}")
        for i, (s, a) in enumerate(zip(salience, active)):
            if a:
                st.write(f"Item {i+1}: {above_water[i]:.2f} / {above_water.sum():.2f} = **{s*100:.1f}%**")
            else:
                st.write(f"Item {i+1}: **0%** (below water)")

with tab3:
    st.subheader("Side-by-Side: Normalization vs Competition")
    
    st.markdown("""
    <div class="key-message">
        🔍 <strong>The Critical Test:</strong> What happens to weak indicators?
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison table
    comparison_data = {
        "Item": [f"Item {i+1}" for i in range(5)],
        "Response": [f"{y:.2f}" for y in working_responses],
        "Simple Norm": [f"{n*100:.1f}%" for n in normalized],
        "Competitive": [f"{s*100:.1f}%" if a else "**0%**" for s, a in zip(salience, active)],
        "Difference": [f"{(s-n)*100:+.1f}%" for s, n in zip(salience, normalized)]
    }
    
    st.table(comparison_data)
    
    # Visual comparison
    fig_compare = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Simple Normalization", "Competitive Salience"),
        shared_yaxes=True
    )
    
    fig_compare.add_trace(
        go.Bar(
            x=[f"Item {i+1}" for i in range(5)],
            y=normalized * 100,
            marker_color='#f39c12',
            name="Normalized",
            text=[f"{n*100:.1f}%" for n in normalized],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    fig_compare.add_trace(
        go.Bar(
            x=[f"Item {i+1}" for i in range(5)],
            y=salience * 100,
            marker_color=['#00d4aa' if a else '#dc3545' for a in active],
            name="Competitive",
            text=[f"{s*100:.1f}%" if a else "OUT" for s, a in zip(salience, active)],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig_compare.update_layout(
        height=400,
        showlegend=False,
        template="plotly_dark"
    )
    fig_compare.update_yaxes(range=[0, 80])
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # The punchline
    if n_active < 5:
        dropped = [i+1 for i, a in enumerate(active) if not a]
        st.error(f"""
        **The difference is now visible:**
        - Normalization gives Items {dropped} small but nonzero weights
        - Competition gives them **exactly zero** — they're out of the measurement
        """)
    else:
        st.info("""
        **Currently they look similar.** Raise the water level to see the divergence.
        When indicators drop out, competition and normalization produce fundamentally different results.
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption("""
**Key Insight**: Salience exists because reconstruction exists. The simplex constraint 
forces indicators to compete for limited explanatory capacity. This is structural, 
not just arithmetic rescaling.
""")