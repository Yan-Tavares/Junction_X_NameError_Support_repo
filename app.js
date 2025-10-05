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
        audioPlayerContainer.classList.remove('hidden');
        
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
    loadingIndicator.classList.remove('hidden');

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
        loadingIndicator.classList.add('hidden');
    }
}

// Display results
function displayResults(results) {
    welcomeMessage.classList.add('hidden');
    resultsContainer.classList.remove('hidden');

    // Overall result
    displayOverallResult(results);
    
    // Transcript
    displayTranscript(results);
    
    // Segments
    displaySegments(results);
    
    // Emotion analysis
    if (results.emotion_analysis && results.emotion_analysis.time_series) {
        emotionContainer.classList.remove('hidden');
        displayEmotionAnalysis(results.emotion_analysis);
    } else {
        emotionContainer.classList.add('hidden');
    }
}

// Display overall result
function displayOverallResult(results) {
    const overallResult = document.getElementById('overallResult');
    const label = results.overall_label || 'unknown';
    const confidence = results.confidence || 0;
    const flaggedCount = results.flagged_count || 0;
    const totalSegments = results.total_segments || 0;

    let emoji, message, description, resultClass;

    if (label === 'safe') {
        emoji = '✅';
        message = 'No, this content is safe';
        description = 'No significant extremist speech detected.';
        resultClass = 'result-box-safe';
    } else if (label === 'hate_detected') {
        emoji = '⚠️';
        message = 'Yes, there is!';
        description = 'Extremist or hate speech identified in audio.';
        resultClass = 'result-box-danger';
    } else {
        emoji = '⚠️';
        message = "I'm not 100% certain";
        description = 'Some segments require review. Take a look at the transcript and see if you agree.';
        resultClass = 'result-box-warning';
    }

    overallResult.innerHTML = `
        <div class="result-box ${resultClass}">
            <div class="result-emoji">${emoji}</div>
            <h2 class="result-title">${message}</h2>
            <p class="result-description">${description}</p>
            <div class="result-stats">
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
        transcriptDiv.innerHTML = '<p class="empty-message">No transcript available</p>';
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
        segmentsList.innerHTML = '<p class="empty-message">✅ No segments exceeded the threshold</p>';
        return;
    }

    segmentsList.innerHTML = displaySegments.map((segment, i) => {
        const label = segment.label || 'unknown';
        const confidence = segment.confidence || 0;
        const start = segment.start || 0;
        const end = segment.end || 0;
        const text = segment.text || '';

        let icon, segmentClass, badgeClass;
        if (label === 'hate') {
            icon = '🚨';
            segmentClass = 'segment-detail-hate';
            badgeClass = 'badge-danger';
        } else if (label === 'uncertain') {
            icon = '⚠️';
            segmentClass = 'segment-detail-uncertain';
            badgeClass = 'badge-warning';
        } else {
            icon = 'ℹ️';
            segmentClass = 'segment-detail-safe';
            badgeClass = 'badge-success';
        }

        return `
            <details class="segment-detail ${segmentClass}">
                <summary class="segment-summary">
                    <span>${icon}</span>
                    <span class="segment-summary-flex">Segment ${i+1}: ${start.toFixed(1)}s - ${end.toFixed(1)}s</span>
                    <span class="badge ${badgeClass}">${label}</span>
                </summary>
                <div class="segment-content">
                    <p class="segment-text">
                        <strong class="label-strong">Text:</strong> ${escapeHtml(text)}
                    </p>
                    <p class="segment-label-row">
                        <strong class="label-strong">Label:</strong> 
                        <span class="badge ${badgeClass}">${label}</span>
                    </p>
                    <p class="segment-confidence-row">
                        <strong class="label-strong">Confidence:</strong> 
                        <span class="confidence-value">${(confidence * 100).toFixed(1)}%</span>
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
    const emotionChart = document.getElementById('emotionChart');
    const emotionPeaks = document.getElementById('emotionPeaks');

    const timeSeries = emotionAnalysis.time_series || [];
    const peaks = emotionAnalysis.peaks || [];

    if (timeSeries.length === 0) {
        emotionChart.innerHTML = '<p class="empty-message">No emotion data available</p>';
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
            <h3 class="heading-3 text-gray-900 mb-md">🔔 Detected Emotion Peaks:</h3>
            ${peaks.map((peak, i) => `
                <details class="emotion-peak-detail">
                    <summary class="emotion-peak-summary">
                        Peak ${i+1} at ${peak.time.toFixed(1)}s (Arousal: ${peak.arousal.toFixed(2)})
                    </summary>
                    <div class="emotion-peak-content">
                        <p class="emotion-peak-row"><strong class="label-strong">Arousal:</strong> <span class="confidence-value">${peak.arousal.toFixed(2)}</span></p>
                        ${peak.coincides_with ? `<p class="emotion-peak-row"><strong class="label-strong">Coincides with:</strong> <span class="badge badge-warning">${peak.coincides_with}</span></p>` : ''}
                        ${peak.text && peak.text !== 'N/A' ? `<p class="emotion-peak-text"><strong class="label-strong">Text:</strong> ${escapeHtml(peak.text)}</p>` : ''}
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
