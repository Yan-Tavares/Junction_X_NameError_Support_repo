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
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from src.pipeline import VocalFirewallAnalyzer
from src.config import settings

# Helper function to format analyzer results for UI
def format_results_for_ui(results):
    """Convert analyzer output to UI-friendly format"""
    hate_count = sum(1 for s in results["segments"] if s["label"] == "hate")
    flagged_segments = [
        s for s in results["segments"]
        if s["label"] in ["hate", "uncertain"]
    ]
    
    if hate_count > 0:
        overall_label = "hate_detected"
        hate_confs = [s["confidence"] for s in results["segments"] if s["label"] == "hate"]
        confidence = sum(hate_confs) / len(hate_confs)
    elif len(flagged_segments) > 0:
        overall_label = "uncertain"
        confidence = sum(s["confidence"] for s in flagged_segments) / len(flagged_segments)
    else:
        overall_label = "safe"
        safe_confs = [s["confidence"] for s in results["segments"] if s["label"] == "non-hate"]
        confidence = sum(safe_confs) / len(safe_confs) if safe_confs else 1.0
    
    return {
        "transcript": results["transcript"],
        "overall_label": overall_label,
        "confidence": confidence,
        "segments": results["segments"],
        "segments": results["segments"],
        "emotion_analysis": results.get("emotion_analysis"),
        "flagged_count": len(flagged_segments),
        "total_segments": len(results["segments"])
    }

def generate_colored_transcript(segments):
    """Generate HTML with color-coded transcript based on segment labels"""
    if not segments:
        return "<p style='color: grey;'>No transcript available</p>"
    
    # Sort segments by start time to ensure proper order
    sorted_spans = sorted(segments, key=lambda x: x.get("start", 0))
    
    html_parts = []
    for segment in sorted_spans:
        text = segment.get("text", "")
        label = segment.get("label", "unknown")
        
        # Determine color based on label
        if label == "hate":
            color = "#dc3545"  # Red
        elif label == "uncertain":
            color = "#ffc107"  # Orange/yellow
        else:  # non-hate or safe
            color = "#fafafa"  # Grey
        
        # Add colored text with a space separator
        html_parts.append(f'<span style="color: {color};">{text}</span>')
    
    # Join with spaces
    return " ".join(html_parts)

# Page configuration
st.set_page_config(
    page_title="Safe Speech Checker",
    page_icon="🗣️", # person speaking
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

# Header
st.markdown('<h1 class="main-header">🗣️ Is there extremist speech in this audio?</h1>', unsafe_allow_html=True)

# Sidebar with upload and settings
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    An automated system for detecting extremist speech in audio files, providing timestamped analysis of potentially harmful content.
    """)
    
    st.divider()
    
    # Settings section - must come first so variables are defined
    with st.expander("⚙️ Settings", expanded=False):
        api_url = st.text_input(
            "API Endpoint", 
            value=os.getenv("API_URL", "http://localhost:8000"),
            help="Backend API server URL"
        )
        
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
    
    st.divider()
    
    # Upload section in sidebar
    st.header("📤 Upload Audio")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'],
        help="Upload audio file for analysis",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Display audio player
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')

        # Analyze button
        if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{api_url}/analyze", files=files, timeout=300)
                    
                    if response.status_code == 200:
                        st.session_state.analysis_results = response.json()
                        st.success("✅ Complete!")
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {response.text}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Upload an audio file to begin")

# Main content - full width for results
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Overall assessment
    overall_label = results.get("overall_label", "unknown")
    confidence = results.get("confidence", 0.0)
    flagged_count = results.get("flagged_count", 0)
    total_segments = results.get("total_segments", 0)

    if overall_label == "safe":
        emoji = "✅"
        message = "No, this content is safe"
        md_inner = f"""
        <div style="font-size: 1.2rem;">
            <p>No significant extremist speech detected.</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Segments:</strong> {flagged_count}/{total_segments} flagged</p>
        </div>
        """
    elif overall_label == "hate_detected":
        emoji = "⚠️"
        message = "Yes, there is!"
        md_inner = f"""
        <div style="font-size: 1.2rem;">
            <p>Extremist or hate speech identified in audio.</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Segments:</strong> {flagged_count}/{total_segments} flagged</p>
        </div>
        """
    else:
        emoji = "⚠️"
        message = "I'm not 100% certain"
        md_inner = f"""
        <div style="font-size: 1.2rem;">
            <p>Some segments require review. Take a look at the transcript and see if you agree.</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Segments:</strong> {flagged_count}/{total_segments} flagged</p>
        </div>
        """
    
    st.subheader(emoji + " Result: " + message)

    st.markdown(md_inner, unsafe_allow_html=True)
    
    st.write("")

    # Button to export results as JSON
    results_json = json.dumps(results, indent=2)
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
        label="📥 Export Results as JSON",
        data=results_json,
        file_name=f"analysis_{st.session_state.uploaded_file_name}.json",
        mime="application/json",
        use_container_width=True
    )

    st.write("")

    # Transcript with color coding
    st.subheader("📝 Transcript")
    segments = results.get("segments", results.get("segments", []))
    if segments:
        colored_html = generate_colored_transcript(segments)
        st.markdown(
            f'<div style="background-color: #26272f; padding: 1rem; border-radius: 0.5rem; line-height: 1.8; font-size: 1rem;">{colored_html}</div>',
            unsafe_allow_html=True
        )
        st.caption("<p style='color: #ffffff;'>🔴 Red = Hate speech | 🟡 Orange = Uncertain | ⚪ Grey = Safe</p>", unsafe_allow_html=True)
    else:
        transcript = results.get("transcript", "No transcript available")
        st.text_area("Full Transcript", transcript, height=150, disabled=True)
    
    # Hate spans/segments
    segments = results.get("segments", results.get("segments", []))
    if segments:
        st.subheader("🔍 Analyzed Segments")
        
        # Filter for display based on threshold
        display_segments = [
            s for s in segments
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
else:
    st.info("👈 Upload and analyze an audio file to see results")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built for Junction X Hackathon | Team NameError</p>
</div>
""", unsafe_allow_html=True)

