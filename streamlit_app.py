"""
Vocal Firewall - Streamlit UI
Detects extremist speech in audio files and provides timestamped results.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import time
import os
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from src.pipeline import VocalFirewallAnalyzer
from src.config import settings

# Helper function to format analyzer results for UI
def format_results_for_ui(results):
    """Convert analyzer output to UI-friendly format"""
    hate_count = sum(1 for s in results["hate_spans"] if s["label"] == "hate")
    flagged_segments = [
        s for s in results["hate_spans"]
        if s["label"] in ["hate", "uncertain"]
    ]
    
    if hate_count > 0:
        overall_label = "hate_detected"
        hate_confs = [s["confidence"] for s in results["hate_spans"] if s["label"] == "hate"]
        confidence = sum(hate_confs) / len(hate_confs)
    elif len(flagged_segments) > 0:
        overall_label = "uncertain"
        confidence = sum(s["confidence"] for s in flagged_segments) / len(flagged_segments)
    else:
        overall_label = "safe"
        safe_confs = [s["confidence"] for s in results["hate_spans"] if s["label"] == "non-hate"]
        confidence = sum(safe_confs) / len(safe_confs) if safe_confs else 1.0
    
    return {
        "transcript": results["transcript"],
        "overall_label": overall_label,
        "confidence": confidence,
        "hate_spans": results["hate_spans"],
        "segments": results["segments"],
        "emotion_analysis": results.get("emotion_analysis"),
        "flagged_count": len(flagged_segments),
        "total_segments": len(results["hate_spans"])
    }

# Page configuration
st.set_page_config(
    page_title="Vocal Firewall - Extremist Speech Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-box {
        background-color: #22382a;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'use_local' not in st.session_state:
    st.session_state.use_local = False

# Header
st.markdown('<h1 class="main-header">🛡️ Vocal Firewall</h1>', unsafe_allow_html=True)
st.markdown("### Automated Detection of Extremist Speech in Audio")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system analyzes audio files to detect:
    - **Derogatory** language
    - **Exclusionary** rhetoric
    - **Dangerous** speech
    
    **How it works:**
    1. Upload an audio file
    2. System transcribes speech
    3. AI analyzes content
    4. Get timestamped results
    """)
    
    st.header("⚙️ Settings")
    
    # Mode selection
    mode = st.radio(
        "Processing Mode",
        ["API Server", "Local (Direct)"],
        help="API Server: Uses FastAPI backend | Local: Direct analysis"
    )
    st.session_state.use_local = (mode == "Local (Direct)")
    
    if not st.session_state.use_local:
        api_url = st.text_input(
            "API Endpoint", 
            value=os.getenv("API_URL", "http://localhost:8000"),
            help="Backend API server URL"
        )
    else:
        api_url = None
        # Initialize local analyzer if needed
        if st.session_state.analyzer is None:
            with st.spinner("Loading models..."):
                try:
                    st.session_state.analyzer = VocalFirewallAnalyzer(
                        whisper_model_size=settings.WHISPER_MODEL_SIZE,
                        text_model_path=settings.TEXT_MODEL_PATH,
                        enable_emotion=settings.ENABLE_EMOTION_ANALYSIS,
                        fast_mode=settings.FAST_MODE
                    )
                    st.success("✅ Models loaded!")
                except Exception as e:
                    st.error(f"❌ Failed to load models: {e}")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence to flag content"
    )
    
    enable_emotion = st.checkbox(
        "Enable Emotion Analysis",
        value=settings.ENABLE_EMOTION_ANALYSIS,
        help="Analyze speech emotions (arousal, valence, dominance)"
    )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Audio")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'],
        help="Upload audio file for analysis"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Display audio player
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
        
        # File info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📁 **File:** {uploaded_file.name} ({file_size:.2f} MB)")
        
        # Analyze button
        if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing audio... This may take a moment..."):
                try:
                    if st.session_state.use_local:
                        # Local analysis
                        if st.session_state.analyzer is None:
                            st.error("❌ Analyzer not initialized!")
                        else:
                            # Save uploaded file temporarily
                            temp_path = settings.TEMP_DIR / uploaded_file.name
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            
                            # Run analysis
                            results = st.session_state.analyzer.analyze_audio(temp_path)
                            
                            # Clean up temp file
                            temp_path.unlink()
                            
                            # Format results for UI
                            st.session_state.analysis_results = format_results_for_ui(results)
                            st.success("✅ Analysis complete!")
                    else:
                        # API-based analysis
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{api_url}/analyze", files=files, timeout=300)
                        
                        if response.status_code == 200:
                            st.session_state.analysis_results = response.json()
                            st.success("✅ Analysis complete!")
                        else:
                            st.error(f"❌ Error: {response.text}")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

with col2:
    st.header("📊 Results")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Overall assessment
        overall_label = results.get("overall_label", "unknown")
        confidence = results.get("confidence", 0.0)
        flagged_count = results.get("flagged_count", 0)
        total_segments = results.get("total_segments", 0)
        
        if overall_label == "safe":
            st.markdown(f"""
            <div class="safe-box">
                <h3>✅ Safe Content</h3>
                <p>No significant extremist speech detected.</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Segments:</strong> {flagged_count}/{total_segments} flagged</p>
            </div>
            """, unsafe_allow_html=True)
        elif overall_label == "hate_detected":
            st.markdown(f"""
            <div class="danger-box">
                <h3>⚠️ Hate Speech Detected</h3>
                <p>Extremist or hate speech identified in audio.</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Segments:</strong> {flagged_count}/{total_segments} flagged</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ Uncertain Content</h3>
                <p>Some segments require review.</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Segments:</strong> {flagged_count}/{total_segments} flagged</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Transcript
        st.subheader("📝 Transcript")
        transcript = results.get("transcript", "No transcript available")
        st.text_area("Full Transcript", transcript, height=150, disabled=True)
        
        # Hate spans/segments
        hate_spans = results.get("hate_spans", results.get("segments", []))
        if hate_spans:
            st.subheader("🔍 Analyzed Segments")
            
            # Filter for display based on threshold
            display_segments = [
                s for s in hate_spans
                if s.get("label") in ["hate", "uncertain"] or 
                   (s.get("label") == "hate" and s.get("confidence", 0) > confidence_threshold)
            ]
            
            if display_segments:
                for i, segment in enumerate(display_segments):
                    label = segment.get("label", "unknown")
                    conf = segment.get("confidence", 0.0)
                    
                    # Format dict or object
                    if isinstance(segment, dict):
                        start = segment.get("start", 0)
                        end = segment.get("end", 0)
                        text = segment.get("text", "")
                    else:
                        start = getattr(segment, "start", 0)
                        end = getattr(segment, "end", 0)
                        text = getattr(segment, "text", "")
                        label = getattr(segment, "label", "unknown")
                        conf = getattr(segment, "confidence", 0.0)
                    
                    # Color code by label
                    if label == "hate":
                        icon = "🚨"
                        color = "red"
                    elif label == "uncertain":
                        icon = "⚠️"
                        color = "orange"
                    else:
                        icon = "ℹ️"
                        color = "blue"
                    
                    with st.expander(f"{icon} Segment {i+1}: {start:.1f}s - {end:.1f}s ({label})"):
                        st.write(f"**Text:** {text}")
                        st.write(f"**Label:** :{color}[{label.upper()}]")
                        st.write(f"**Confidence:** {conf:.1%}")
                        
                        # Progress bar for confidence
                        st.progress(conf)
            else:
                st.info("✅ No segments exceeded the threshold")
        
        # Emotion Analysis Visualization
        emotion_analysis = results.get("emotion_analysis")
        if emotion_analysis and emotion_analysis.get("time_series"):
            st.subheader("😊 Emotion Analysis")
            
            time_series = emotion_analysis.get("time_series", [])
            peaks = emotion_analysis.get("peaks", [])
            
            if time_series:
                # Extract data for plotting
                times = [ts["time"] for ts in time_series]
                arousal = [ts["arousal"] for ts in time_series]
                valence = [ts["valence"] for ts in time_series]
                dominance = [ts["dominance"] for ts in time_series]
                
                # Create subplot figure
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("Arousal", "Valence", "Dominance"),
                    shared_xaxes=True,
                    vertical_spacing=0.1
                )
                
                # Arousal plot
                fig.add_trace(
                    go.Scatter(x=times, y=arousal, name="Arousal", 
                              line=dict(color="red", width=2)),
                    row=1, col=1
                )
                
                # Add peak markers
                if peaks:
                    peak_times = [p["time"] for p in peaks if isinstance(p, dict)]
                    peak_arousal = [p["arousal"] for p in peaks if isinstance(p, dict)]
                    fig.add_trace(
                        go.Scatter(x=peak_times, y=peak_arousal, 
                                  mode="markers", name="Arousal Peaks",
                                  marker=dict(size=10, color="darkred", symbol="diamond")),
                        row=1, col=1
                    )
                
                # Valence plot
                fig.add_trace(
                    go.Scatter(x=times, y=valence, name="Valence",
                              line=dict(color="green", width=2)),
                    row=2, col=1
                )
                
                # Dominance plot
                fig.add_trace(
                    go.Scatter(x=times, y=dominance, name="Dominance",
                              line=dict(color="blue", width=2)),
                    row=3, col=1
                )
                
                fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
                fig.update_yaxes(title_text="Score", range=[-1, 1])
                fig.update_layout(height=600, showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show emotion peaks
                if peaks:
                    st.write("**🔔 Detected Emotion Peaks:**")
                    for i, peak in enumerate(peaks):
                        if isinstance(peak, dict):
                            peak_time = peak.get("time", 0)
                            peak_arousal = peak.get("arousal", 0)
                            coincides = peak.get("coincides_with", None)
                            peak_text = peak.get("text", "N/A")
                            
                            with st.expander(f"Peak {i+1} at {peak_time:.1f}s (Arousal: {peak_arousal:.2f})"):
                                st.write(f"**Arousal:** {peak_arousal:.2f}")
                                if coincides:
                                    st.write(f"**Coincides with:** {coincides}")
                                if peak_text and peak_text != "N/A":
                                    st.write(f"**Text:** {peak_text}")
        
        # Export option
        st.subheader("💾 Export Results")
        if st.button("Download Results (JSON)", use_container_width=True):
            import json
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=results_json,
                file_name=f"analysis_{st.session_state.uploaded_file_name}.json",
                mime="application/json"
            )
    else:
        st.info("👈 Upload and analyze an audio file to see results")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built for Junction X Hackathon | Team NameError</p>
    <p><small>⚠️ This is a prototype system for demonstration purposes</small></p>
</div>
""", unsafe_allow_html=True)

