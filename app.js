// State management
let analysisResults = null;
let uploadedFile = null;
let youtubeUrl = null;
let youtubeVideoId = null;
let currentInputMode = 'file'; // 'file' or 'youtube'
let ytPlayer = null;
let syncInterval = null;

// DOM Elements
const audioFileInput = document.getElementById('audioFile');
const audioPlayer = document.getElementById('audioPlayer');
const audioPlayerContainer = document.getElementById('audioPlayerContainer');
const uploadDropzone = document.getElementById('uploadDropzone');
const fileInfo = document.getElementById('fileInfo');
const fileInfoName = document.getElementById('fileInfoName');
const fileInfoSize = document.getElementById('fileInfoSize');
const removeFileBtn = document.getElementById('removeFileBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const welcomeMessage = document.getElementById('welcomeMessage');
const resultsContainer = document.getElementById('resultsContainer');
const emotionContainer = document.getElementById('emotionContainer');
const apiUrlInput = document.getElementById('apiUrl');
const confidenceThresholdInput = document.getElementById('confidenceThreshold');
const thresholdValue = document.getElementById('thresholdValue');
const exportBtn = document.getElementById('exportBtn');

// Input mode toggle elements
const fileModeBtn = document.getElementById('fileModeBtn');
const youtubeModeBtn = document.getElementById('youtubeModeBtn');
const fileUploadSection = document.getElementById('fileUploadSection');
const youtubeSection = document.getElementById('youtubeSection');

// YouTube elements
const youtubeUrlInput = document.getElementById('youtubeUrl');
const loadYoutubeBtn = document.getElementById('loadYoutubeBtn');
const youtubePlayerContainer = document.getElementById('youtubePlayerContainer');
const youtubePlayer = document.getElementById('youtubePlayer');
const youtubeMainPlayerContainer = document.getElementById('youtubeMainPlayerContainer');
const youtubeMainPlayer = document.getElementById('youtubeMainPlayer');

// Load YouTube IFrame API
let youtubeAPIReady = false;
const tag = document.createElement('script');
tag.src = 'https://www.youtube.com/iframe_api';
const firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

// YouTube API ready callback
window.onYouTubeIframeAPIReady = function() {
    youtubeAPIReady = true;
};

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

// Drag and drop events
uploadDropzone.addEventListener('click', () => audioFileInput.click());
uploadDropzone.addEventListener('dragover', handleDragOver);
uploadDropzone.addEventListener('dragleave', handleDragLeave);
uploadDropzone.addEventListener('drop', handleDrop);
removeFileBtn.addEventListener('click', handleRemoveFile);

// Input mode toggle events
fileModeBtn.addEventListener('click', () => switchInputMode('file'));
youtubeModeBtn.addEventListener('click', () => switchInputMode('youtube'));

// YouTube events
loadYoutubeBtn.addEventListener('click', loadYoutubeVideo);

// Remove focus flash from details/summary elements
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('details summary').forEach(summary => {
        summary.addEventListener('mousedown', (e) => {
            e.preventDefault();
            summary.blur();
        });
        summary.addEventListener('click', (e) => {
            // Allow the default toggle behavior
            setTimeout(() => summary.blur(), 0);
        });
    });
});

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadDropzone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadDropzone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadDropzone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        // Check if it's an audio file
        if (file.type.startsWith('audio/') || 
            /\.(wav|mp3|ogg|flac|m4a|webm)$/i.test(file.name)) {
            processFile(file);
        } else {
            alert('Please upload an audio file (WAV, MP3, OGG, FLAC, M4A, WebM)');
        }
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Process selected file
function processFile(file) {
    uploadedFile = file;
    
    // Hide dropzone, show file info
    uploadDropzone.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    
    // Update file info
    fileInfoName.textContent = file.name;
    fileInfoSize.textContent = formatFileSize(file.size);
    
    // Set up audio player
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    audioPlayerContainer.classList.remove('hidden');
    
    // Enable analyze button
    analyzeBtn.disabled = false;
}

// Handle file removal
function handleRemoveFile(e) {
    e.stopPropagation();
    
    // Clear file
    uploadedFile = null;
    audioFileInput.value = '';
    
    // Show dropzone, hide file info
    uploadDropzone.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    
    // Hide audio player
    audioPlayerContainer.classList.add('hidden');
    audioPlayer.src = '';
    
    // Disable analyze button
    analyzeBtn.disabled = true;
}

// Switch input mode between file and YouTube
function switchInputMode(mode) {
    currentInputMode = mode;
    
    if (mode === 'file') {
        // Show file section, hide YouTube section
        fileUploadSection.classList.remove('hidden');
        youtubeSection.classList.add('hidden');
        
        // Update button styles
        fileModeBtn.classList.add('active');
        youtubeModeBtn.classList.remove('active');
        
        // Enable analyze if file is uploaded
        analyzeBtn.disabled = !uploadedFile;
    } else if (mode === 'youtube') {
        // Show YouTube section, hide file section
        fileUploadSection.classList.add('hidden');
        youtubeSection.classList.remove('hidden');
        
        // Update button styles
        fileModeBtn.classList.remove('active');
        youtubeModeBtn.classList.add('active');
        
        // Enable analyze if YouTube video is loaded
        analyzeBtn.disabled = !youtubeUrl;
    }
}

// Extract YouTube video ID from URL
function extractYoutubeVideoId(url) {
    const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[7].length === 11) ? match[7] : null;
}

// Load YouTube video
function loadYoutubeVideo() {
    const url = youtubeUrlInput.value.trim();
    
    if (!url) {
        alert('Please enter a YouTube URL');
        return;
    }
    
    const videoId = extractYoutubeVideoId(url);
    
    if (!videoId) {
        alert('Invalid YouTube URL. Please enter a valid YouTube video link.');
        return;
    }
    
    // Store the URL and video ID
    youtubeUrl = url;
    youtubeVideoId = videoId;
    
    // Embed the video
    youtubePlayer.src = `https://www.youtube.com/embed/${videoId}`;
    youtubePlayerContainer.classList.remove('hidden');
    
    // Enable analyze button
    analyzeBtn.disabled = false;
}

// Utility to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Analyze audio
async function analyzeAudio() {
    const apiUrl = apiUrlInput.value.trim();
    if (!apiUrl) {
        alert('Please enter an API endpoint');
        return;
    }

    // Check if we have input
    if (currentInputMode === 'file' && !uploadedFile) {
        alert('Please select an audio file first');
        return;
    }
    
    if (currentInputMode === 'youtube' && !youtubeUrl) {
        alert('Please load a YouTube video first');
        return;
    }

    // Show loading
    analyzeBtn.disabled = true;
    loadingIndicator.classList.remove('hidden');

    try {
        let response;
        
        if (currentInputMode === 'youtube') {
            // YouTube analysis mode
            response = await fetch(`${apiUrl}/analyze/youtube`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: youtubeUrl })
            });
        } else {
            // File upload mode
            const formData = new FormData();
            formData.append('file', uploadedFile);

            response = await fetch(`${apiUrl}/analyze`, {
                method: 'POST',
                body: formData
            });
        }

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

    // If YouTube mode, show main player and initialize sync (do this last)
    if (currentInputMode === 'youtube' && youtubeVideoId) {
        youtubeMainPlayerContainer.classList.remove('hidden');
        // Use setTimeout to ensure DOM is fully updated before initializing player
        setTimeout(() => initializeYoutubePlayer(youtubeVideoId), 100);
    } else {
        youtubeMainPlayerContainer.classList.add('hidden');
        stopTranscriptSync();
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
        emoji = '✓';
        message = 'Content appears safe';
        description = 'No significant extremist speech patterns detected in the analysis.';
        resultClass = 'result-box-safe';
    } else if (label === 'hate_detected') {
        emoji = '!';
        message = 'Concerning content detected';
        description = 'The analysis identified potential extremist or hate speech patterns.';
        resultClass = 'result-box-danger';
    } else {
        emoji = '?';
        message = 'Review recommended';
        description = 'Some segments require manual review. Please examine the transcript and flagged segments below.';
        resultClass = 'result-box-warning';
    }

    overallResult.innerHTML = `
        <div class="result-box ${resultClass}">
            <div class="result-emoji">${emoji}</div>
            <h2 class="result-title">${message}</h2>
            <p class="result-description">${description}</p>
            <div class="result-stats">
                <div><strong>Confidence:</strong> ${(confidence * 100).toFixed(1)}%</div>
                <div><strong>Flagged Segments:</strong> ${flaggedCount} of ${totalSegments}</div>
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

    // Generate colored HTML with data attributes for timing
    const html = sortedSegments.map((segment, index) => {
        const text = segment.text || '';
        const label = segment.label || 'unknown';
        const start = segment.start || 0;
        const end = segment.end || 0;
        
        let className;
        if (label === 'hate') {
            className = 'segment-hate';
        } else if (label === 'uncertain') {
            className = 'segment-uncertain';
        } else {
            className = 'segment-safe';
        }
        
        return `<span class="${className}" data-segment-index="${index}" data-start="${start}" data-end="${end}">${escapeHtml(text)}</span>`;
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
        segmentsList.innerHTML = '<p class="empty-message">No segments exceeded the threshold</p>';
        return;
    }

    segmentsList.innerHTML = displaySegments.map((segment, i) => {
        const label = segment.label || 'unknown';
        const confidence = segment.confidence || 0;
        const start = segment.start || 0;
        const end = segment.end || 0;
        const text = segment.text || '';

        let segmentClass, badgeClass;
        if (label === 'hate') {
            segmentClass = 'segment-detail-hate';
            badgeClass = 'badge-danger';
        } else if (label === 'uncertain') {
            segmentClass = 'segment-detail-uncertain';
            badgeClass = 'badge-warning';
        } else {
            segmentClass = 'segment-detail-safe';
            badgeClass = 'badge-success';
        }

        return `
            <details class="segment-detail ${segmentClass}">
                <summary class="segment-summary">
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
            <h3 class="heading-3 text-gray-900 mb-md">Detected Emotion Peaks</h3>
            ${peaks.map((peak, i) => `
                <details class="emotion-peak-detail">
                    <summary class="emotion-peak-summary">
                        Peak ${i+1} at ${peak.time.toFixed(1)}s (Arousal: ${peak.arousal.toFixed(2)})
                    </summary>
                    <div class="emotion-peak-content">
                        <p class="emotion-peak-row"><strong class="label-strong">Arousal Level:</strong> <span class="confidence-value">${peak.arousal.toFixed(2)}</span></p>
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
    
    // Generate filename based on input mode
    let filename;
    if (currentInputMode === 'file' && uploadedFile) {
        filename = `analysis_${uploadedFile.name}.json`;
    } else if (currentInputMode === 'youtube' && youtubeUrl) {
        const videoId = extractYoutubeVideoId(youtubeUrl);
        filename = `analysis_youtube_${videoId || 'video'}.json`;
    } else {
        filename = 'analysis_results.json';
    }
    
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Initialize YouTube player with IFrame API
function initializeYoutubePlayer(videoId) {
    console.log('Initializing YouTube player with video ID:', videoId);
    
    // Destroy existing player if any
    if (ytPlayer && ytPlayer.destroy) {
        try {
            ytPlayer.destroy();
        } catch (e) {
            console.log('Error destroying previous player:', e);
        }
        ytPlayer = null;
    }
    
    // Stop any existing sync
    stopTranscriptSync();
    
    // Clear the container and ensure it exists
    const container = document.getElementById('youtubeMainPlayer');
    if (!container) {
        console.error('YouTube player container not found');
        return;
    }
    container.innerHTML = '';
    
    // Wait for API to be ready
    const initPlayer = () => {
        try {
            console.log('Creating YouTube player...');
            ytPlayer = new YT.Player('youtubeMainPlayer', {
                height: '100%',
                width: '100%',
                videoId: videoId,
                playerVars: {
                    'enablejsapi': 1,
                    'origin': window.location.origin,
                    'rel': 0,
                    'modestbranding': 1
                },
                events: {
                    'onReady': onPlayerReady,
                    'onStateChange': onPlayerStateChange,
                    'onError': onPlayerError
                }
            });
        } catch (error) {
            console.error('Error creating YouTube player:', error);
            // Fallback: create a simple iframe
            container.innerHTML = `<iframe src="https://www.youtube.com/embed/${videoId}?enablejsapi=1" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen
                style="width: 100%; height: 100%;"></iframe>`;
        }
    };
    
    if (youtubeAPIReady && window.YT && window.YT.Player) {
        initPlayer();
    } else {
        console.log('Waiting for YouTube API to load...');
        // Wait for API to load with timeout
        let attempts = 0;
        const maxAttempts = 50; // 5 seconds
        const checkAPI = setInterval(() => {
            attempts++;
            if (youtubeAPIReady && window.YT && window.YT.Player) {
                console.log('YouTube API ready!');
                clearInterval(checkAPI);
                initPlayer();
            } else if (attempts >= maxAttempts) {
                console.warn('YouTube API timeout, using fallback iframe');
                clearInterval(checkAPI);
                // Fallback: create a simple iframe
                container.innerHTML = `<iframe src="https://www.youtube.com/embed/${videoId}?enablejsapi=1" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen
                    style="width: 100%; height: 100%;"></iframe>`;
            }
        }, 100);
    }
}

// YouTube player ready callback
function onPlayerReady(event) {
    console.log('YouTube player ready');
    // Player is ready, sync will start when playing
}

// YouTube player state change callback
function onPlayerStateChange(event) {
    console.log('Player state changed:', event.data);
    // YT.PlayerState.PLAYING = 1
    if (event.data === 1) {
        startTranscriptSync();
    } else {
        stopTranscriptSync();
    }
}

// YouTube player error callback
function onPlayerError(event) {
    console.error('YouTube player error:', event.data);
    // Common error codes:
    // 2 - invalid parameter
    // 5 - HTML5 player error
    // 100 - video not found
    // 101, 150 - video not allowed to be played in embedded players
}

// Start syncing transcript with video playback
function startTranscriptSync() {
    if (syncInterval) return; // Already running
    
    syncInterval = setInterval(() => {
        if (ytPlayer && ytPlayer.getCurrentTime) {
            const currentTime = ytPlayer.getCurrentTime();
            updateTranscriptHighlight(currentTime);
        }
    }, 100); // Update every 100ms for smooth syncing
}

// Stop transcript sync
function stopTranscriptSync() {
    if (syncInterval) {
        clearInterval(syncInterval);
        syncInterval = null;
    }
    
    // Remove all active highlights
    const transcriptDiv = document.getElementById('transcript');
    if (transcriptDiv) {
        const activeSegments = transcriptDiv.querySelectorAll('.segment-active');
        activeSegments.forEach(seg => seg.classList.remove('segment-active'));
    }
}

// Update transcript highlight based on current time
function updateTranscriptHighlight(currentTime) {
    const transcriptDiv = document.getElementById('transcript');
    if (!transcriptDiv) return;
    
    const segments = transcriptDiv.querySelectorAll('[data-start][data-end]');
    let activeSegment = null;
    let segmentChanged = false;
    
    // Find the segment that contains the current time
    segments.forEach(segment => {
        const start = parseFloat(segment.dataset.start);
        const end = parseFloat(segment.dataset.end);
        
        if (currentTime >= start && currentTime <= end) {
            activeSegment = segment;
            if (!segment.classList.contains('segment-active')) {
                segment.classList.add('segment-active');
                segmentChanged = true;
            }
        } else {
            if (segment.classList.contains('segment-active')) {
                segment.classList.remove('segment-active');
            }
        }
    });
    
    // Auto-scroll to keep active segment in view (Spotify-like)
    if (activeSegment && segmentChanged) {
        // Calculate position to center the active segment
        const containerHeight = transcriptDiv.clientHeight;
        const segmentOffsetTop = activeSegment.offsetTop;
        const segmentHeight = activeSegment.offsetHeight;
        
        // Center the segment vertically in the container
        const targetScrollTop = segmentOffsetTop - (containerHeight / 2) + (segmentHeight / 2);
        
        // Smooth scroll to position
        transcriptDiv.scrollTo({
            top: Math.max(0, targetScrollTop),
            behavior: 'smooth'
        });
    }
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
