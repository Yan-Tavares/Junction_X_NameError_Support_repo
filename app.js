// State management
let analysisResults = null;
let uploadedFile = null;

// DOM Elements
const audioFileInput = document.getElementById('audioFile');
const audioPlayer = document.getElementById('audioPlayer');
const audioPlayerContainer = document.getElementById('audioPlayerContainer');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const welcomeMessage = document.getElementById('welcomeMessage');
const resultsContainer = document.getElementById('resultsContainer');
const emotionContainer = document.getElementById('emotionContainer');
const apiUrlInput = document.getElementById('apiUrl');
const confidenceThresholdInput = document.getElementById('confidenceThreshold');
const thresholdValue = document.getElementById('thresholdValue');
const exportBtn = document.getElementById('exportBtn');

// Event Listeners
audioFileInput.addEventListener('change', handleFileSelect);
analyzeBtn.addEventListener('click', analyzeAudio);
confidenceThresholdInput.addEventListener('input', (e) => {
    thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
    if (analysisResults) {
        displayResults(analysisResults);
    }
});
exportBtn.addEventListener('click', exportResults);

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        uploadedFile = file;
        fileName.textContent = file.name;
        
        // Set up audio player
        const url = URL.createObjectURL(file);
        audioPlayer.src = url;
        audioPlayerContainer.style.display = 'block';
        
        // Enable analyze button
        analyzeBtn.disabled = false;
    }
}

// Analyze audio
async function analyzeAudio() {
    if (!uploadedFile) {
        alert('Please select an audio file first');
        return;
    }

    const apiUrl = apiUrlInput.value.trim();
    if (!apiUrl) {
        alert('Please enter an API endpoint');
        return;
    }

    // Show loading
    analyzeBtn.disabled = true;
    loadingIndicator.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);

        const response = await fetch(`${apiUrl}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API Error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        analysisResults = data;
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        alert(`Error analyzing audio: ${error.message}`);
        console.error('Analysis error:', error);
    } finally {
        analyzeBtn.disabled = false;
        loadingIndicator.style.display = 'none';
    }
}

// Display results
function displayResults(results) {
    welcomeMessage.style.display = 'none';
    resultsContainer.style.display = 'block';

    // Overall result
    displayOverallResult(results);
    
    // Transcript
    displayTranscript(results);
    
    // Segments
    displaySegments(results);
    
    // Emotion analysis
    if (results.emotion_analysis && results.emotion_analysis.time_series) {
        emotionContainer.style.display = 'block';
        displayEmotionAnalysis(results.emotion_analysis);
    } else {
        emotionContainer.style.display = 'none';
    }
}

// Display overall result
function displayOverallResult(results) {
    const overallResult = document.getElementById('overallResult');
    const label = results.overall_label || 'unknown';
    const confidence = results.confidence || 0;
    const flaggedCount = results.flagged_count || 0;
    const totalSegments = results.total_segments || 0;

    let emoji, message, description, bgColor, borderColor;

    if (label === 'safe') {
        emoji = '✅';
        message = 'No, this content is safe';
        description = 'No significant extremist speech detected.';
        bgColor = 'var(--success-light)';
        borderColor = 'var(--success)';
    } else if (label === 'hate_detected') {
        emoji = '⚠️';
        message = 'Yes, there is!';
        description = 'Extremist or hate speech identified in audio.';
        bgColor = 'var(--danger-light)';
        borderColor = 'var(--danger)';
    } else {
        emoji = '⚠️';
        message = "I'm not 100% certain";
        description = 'Some segments require review. Take a look at the transcript and see if you agree.';
        bgColor = 'var(--warning-light)';
        borderColor = 'var(--warning)';
    }

    overallResult.innerHTML = `
        <div style="text-align: center; background: ${bgColor}; border: 3px solid ${borderColor}; border-radius: 12px; padding: 32px;">
            <div style="font-size: 64px; margin-bottom: 16px;">${emoji}</div>
            <h2 style="font-size: 2rem; font-weight: 700; color: var(--gray-900); margin-bottom: 12px;">${message}</h2>
            <p style="font-size: 1.125rem; color: var(--gray-700); margin-bottom: 24px;">${description}</p>
            <div style="display: flex; justify-content: center; gap: 40px; font-size: 14px; color: var(--gray-600);">
                <div><strong>Confidence:</strong> ${(confidence * 100).toFixed(1)}%</div>
                <div><strong>Segments:</strong> ${flaggedCount}/${totalSegments} flagged</div>
            </div>
        </div>
    `;
}

// Display colored transcript
function displayTranscript(results) {
    const transcriptDiv = document.getElementById('transcript');
    const segments = results.segments || [];

    if (segments.length === 0) {
        transcriptDiv.innerHTML = '<p class="text-gray-500">No transcript available</p>';
        return;
    }

    // Sort segments by start time
    const sortedSegments = [...segments].sort((a, b) => (a.start || 0) - (b.start || 0));

    // Generate colored HTML
    const html = sortedSegments.map(segment => {
        const text = segment.text || '';
        const label = segment.label || 'unknown';
        
        let className;
        if (label === 'hate') {
            className = 'segment-hate';
        } else if (label === 'uncertain') {
            className = 'segment-uncertain';
        } else {
            className = 'segment-safe';
        }
        
        return `<span class="${className}">${escapeHtml(text)}</span>`;
    }).join(' ');

    transcriptDiv.innerHTML = html;
}

// Display segments
function displaySegments(results) {
    const segmentsList = document.getElementById('segmentsList');
    const segments = results.segments || [];
    const threshold = parseFloat(confidenceThresholdInput.value);

    // Filter segments to display
    const displaySegments = segments.filter(s => 
        s.label === 'hate' || 
        s.label === 'uncertain' || 
        (s.label === 'hate' && (s.confidence || 0) > threshold)
    );

    if (displaySegments.length === 0) {
        segmentsList.innerHTML = '<p style="color: var(--gray-600); padding: 20px; text-align: center; background: var(--gray-50); border-radius: 8px;">✅ No segments exceeded the threshold</p>';
        return;
    }

    segmentsList.innerHTML = displaySegments.map((segment, i) => {
        const label = segment.label || 'unknown';
        const confidence = segment.confidence || 0;
        const start = segment.start || 0;
        const end = segment.end || 0;
        const text = segment.text || '';

        let icon, borderColor, badgeClass;
        if (label === 'hate') {
            icon = '🚨';
            borderColor = 'var(--danger)';
            badgeClass = 'badge-danger';
        } else if (label === 'uncertain') {
            icon = '⚠️';
            borderColor = 'var(--warning)';
            badgeClass = 'badge-warning';
        } else {
            icon = 'ℹ️';
            borderColor = 'var(--primary)';
            badgeClass = 'badge-success';
        }

        return `
            <details style="border: 2px solid ${borderColor}; border-radius: 10px; margin-bottom: 16px; overflow: hidden; background: white;">
                <summary style="padding: 16px; font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 8px; background: var(--gray-50);">
                    <span>${icon}</span>
                    <span style="flex: 1;">Segment ${i+1}: ${start.toFixed(1)}s - ${end.toFixed(1)}s</span>
                    <span class="badge ${badgeClass}">${label}</span>
                </summary>
                <div style="padding: 20px; background: white;">
                    <p style="margin-bottom: 12px; color: var(--gray-700); line-height: 1.6;">
                        <strong style="color: var(--gray-900);">Text:</strong> ${escapeHtml(text)}
                    </p>
                    <p style="margin-bottom: 12px;">
                        <strong style="color: var(--gray-900);">Label:</strong> 
                        <span class="badge ${badgeClass}">${label}</span>
                    </p>
                    <p style="margin-bottom: 8px;">
                        <strong style="color: var(--gray-900);">Confidence:</strong> 
                        <span style="color: var(--primary); font-weight: 600;">${(confidence * 100).toFixed(1)}%</span>
                    </p>
                    <div class="progress-bar">
                        <div class="progress-bar-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                </div>
            </details>
        `;
    }).join('');
}

// Display emotion analysis
function displayEmotionAnalysis(emotionAnalysis) {
    const emotionContainer = document.getElementById('emotionContainer');
    const emotionChart = document.getElementById('emotionChart');
    const emotionPeaks = document.getElementById('emotionPeaks');

    emotionContainer.classList.remove('hidden');

    const timeSeries = emotionAnalysis.time_series || [];
    const peaks = emotionAnalysis.peaks || [];

    if (timeSeries.length === 0) {
        emotionChart.innerHTML = '<p class="text-gray-600">No emotion data available</p>';
        return;
    }

    // Extract data
    const times = timeSeries.map(ts => ts.time);
    const arousal = timeSeries.map(ts => ts.arousal);
    const valence = timeSeries.map(ts => ts.valence);
    const dominance = timeSeries.map(ts => ts.dominance);

    // Create traces
    const traces = [
        {
            x: times,
            y: arousal,
            name: 'Arousal',
            type: 'scatter',
            line: { color: 'red', width: 2 },
            yaxis: 'y1'
        },
        {
            x: times,
            y: valence,
            name: 'Valence',
            type: 'scatter',
            line: { color: 'green', width: 2 },
            yaxis: 'y2'
        },
        {
            x: times,
            y: dominance,
            name: 'Dominance',
            type: 'scatter',
            line: { color: 'blue', width: 2 },
            yaxis: 'y3'
        }
    ];

    // Add peak markers
    if (peaks.length > 0) {
        const peakTimes = peaks.map(p => p.time);
        const peakArousal = peaks.map(p => p.arousal);
        traces.push({
            x: peakTimes,
            y: peakArousal,
            name: 'Arousal Peaks',
            mode: 'markers',
            marker: { size: 10, color: 'darkred', symbol: 'diamond' },
            yaxis: 'y1'
        });
    }

    // Layout with subplots
    const layout = {
        grid: { rows: 3, columns: 1, pattern: 'independent' },
        xaxis: { title: 'Time (seconds)' },
        yaxis: { title: 'Arousal', range: [-1, 1] },
        yaxis2: { title: 'Valence', range: [-1, 1] },
        yaxis3: { title: 'Dominance', range: [-1, 1] },
        height: 600,
        showlegend: true
    };

    Plotly.newPlot(emotionChart, traces, layout, { responsive: true });

    // Display peaks
    if (peaks.length > 0) {
        emotionPeaks.innerHTML = `
            <h3 style="font-weight: 700; color: var(--gray-900); margin-bottom: 16px; font-size: 1.25rem;">🔔 Detected Emotion Peaks:</h3>
            ${peaks.map((peak, i) => `
                <details style="border: 2px solid var(--gray-300); border-radius: 10px; margin-bottom: 12px; overflow: hidden; background: white;">
                    <summary style="padding: 14px 16px; font-weight: 600; font-size: 14px; background: var(--gray-50);">
                        Peak ${i+1} at ${peak.time.toFixed(1)}s (Arousal: ${peak.arousal.toFixed(2)})
                    </summary>
                    <div style="padding: 16px; background: white;">
                        <p style="margin-bottom: 8px;"><strong style="color: var(--gray-900);">Arousal:</strong> <span style="color: var(--primary);">${peak.arousal.toFixed(2)}</span></p>
                        ${peak.coincides_with ? `<p style="margin-bottom: 8px;"><strong style="color: var(--gray-900);">Coincides with:</strong> <span class="badge badge-warning">${peak.coincides_with}</span></p>` : ''}
                        ${peak.text && peak.text !== 'N/A' ? `<p style="margin-bottom: 0; color: var(--gray-700); line-height: 1.6;"><strong style="color: var(--gray-900);">Text:</strong> ${escapeHtml(peak.text)}</p>` : ''}
                    </div>
                </details>
            `).join('')}
        `;
    }
}

// Export results as JSON
function exportResults() {
    if (!analysisResults) return;

    const dataStr = JSON.stringify(analysisResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `analysis_${uploadedFile ? uploadedFile.name : 'results'}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
