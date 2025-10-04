"""
Vocal Firewall - Streamlit UI
Detects extremist speech in audio files and provides timestamped results.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import time
import os

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
        background-color: #d4edda;
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
                    # TODO: Replace with actual API call when backend is ready
                    # For MVP, we'll simulate the analysis
                    time.sleep(2)  # Simulate processing time
                    
                    # Mock results for demonstration
                    st.session_state.analysis_results = {
                        "transcript": "This is a sample transcript of the audio file.",
                        "overall_label": "safe",
                        "confidence": 0.92,
                        "categories": {
                            "derogatory": 0.05,
                            "exclusionary": 0.03,
                            "dangerous": 0.02
                        },
                        "flagged_segments": [
                            # Example structure
                            # {"start": 12.5, "end": 15.3, "text": "example text", "label": "derogatory", "confidence": 0.85}
                        ]
                    }
                    
                    st.success("✅ Analysis complete!")
                    
                    # Uncomment when API is ready:
                    # files = {"file": uploaded_file.getvalue()}
                    # response = requests.post(f"{api_url}/analyze", files=files)
                    # if response.status_code == 200:
                    #     st.session_state.analysis_results = response.json()
                    #     st.success("✅ Analysis complete!")
                    # else:
                    #     st.error(f"❌ Error: {response.text}")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

with col2:
    st.header("📊 Results")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Overall assessment
        overall_label = results.get("overall_label", "unknown")
        confidence = results.get("confidence", 0.0)
        
        if overall_label == "safe":
            st.markdown(f"""
            <div class="safe-box">
                <h3>✅ Safe Content</h3>
                <p>No significant extremist speech detected.</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="danger-box">
                <h3>⚠️ Flagged Content</h3>
                <p>Potential extremist speech detected: <strong>{overall_label}</strong></p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Category breakdown
        st.subheader("Category Scores")
        categories = results.get("categories", {})
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(categories.keys()),
                y=list(categories.values()),
                marker_color=['#dc3545' if v > confidence_threshold else '#28a745' for v in categories.values()],
                text=[f"{v:.1%}" for v in categories.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            height=300,
        )
        
        fig.add_hline(y=confidence_threshold, line_dash="dash", line_color="red", 
                     annotation_text="Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transcript
        st.subheader("📝 Transcript")
        transcript = results.get("transcript", "No transcript available")
        st.text_area("Full Transcript", transcript, height=150, disabled=True)
        
        # Flagged segments
        flagged_segments = results.get("flagged_segments", [])
        if flagged_segments:
            st.subheader("⚠️ Flagged Segments")
            
            for i, segment in enumerate(flagged_segments):
                with st.expander(f"Segment {i+1}: {segment['start']:.1f}s - {segment['end']:.1f}s"):
                    st.write(f"**Text:** {segment['text']}")
                    st.write(f"**Category:** {segment['label']}")
                    st.write(f"**Confidence:** {segment['confidence']:.1%}")
        else:
            st.info("No specific segments flagged")
        
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

