# 🖥️ Vocal Firewall CLI

Command-line interface for the Vocal Firewall extremist speech detection API.

**Designed for research purposes** - especially for screening training data for speech models.


## 🆚 CLI vs Web UI

| Feature | CLI | Web UI |
|---------|-----|--------|
| **Target Users** | Researchers | General public |
| **Use Case** | Training data screening | Content review |
| **Batch Processing** | ✅ Yes | ❌ No |
| **CSV Export** | ✅ Yes | ❌ No |
| **Filtering** | ✅ Yes | ❌ No |
| **Automation** | ✅ Yes | ❌ No |
| **Interactive** | ❌ No | ✅ Yes |
| **Visual** | ❌ No | ✅ Yes |
| **Reports** | ✅ Comprehensive | ⚠️ Basic |


## 🚀 Quick Start

```bash
# Make the CLI executable
chmod +x vfw_cli.py

# Check API health
./vfw_cli.py health

# Analyze a single file
./vfw_cli.py analyze audio.wav

# Screen a training data directory
./vfw_cli.py screen ./training_data/
```


## 📋 Prerequisites

1. **API Server Running**: The CLI requires the Vocal Firewall API server to be running
   ```bash
   ./run_api.sh
   ```

2. **Python Dependencies**: Already included in `requirements.txt`
   - requests


## 🔧 Commands

### `health` - Check API Status

Check if the API server is running and models are loaded.

```bash
./vfw_cli.py health

# With custom API URL
./vfw_cli.py --api-url http://192.168.1.100:8000 health
```

### `analyze` - Analyze Single File

Analyze a single audio file for extremist speech.

```bash
# Basic analysis
./vfw_cli.py analyze audio.wav

# Verbose output with segment details
./vfw_cli.py analyze audio.wav -v

# Save results to JSON
./vfw_cli.py analyze audio.wav -o results.json

# Save results to CSV
./vfw_cli.py analyze audio.wav -o results.csv -f csv
```

**Options:**
- `-v, --verbose` - Show detailed segment-by-segment analysis
- `-o, --output FILE` - Save results to file
- `-f, --format {json,csv}` - Output format (default: json)

### `batch` - Batch Process Multiple Files

Analyze multiple audio files efficiently.

**Process specific files:**
```bash
./vfw_cli.py batch file1.wav file2.wav file3.wav
```

**Process entire directory:**
```bash
# Non-recursive
./vfw_cli.py batch -d ./audio_files

# Recursive (scan subdirectories)
./vfw_cli.py batch -d ./audio_files -r

# Save results
./vfw_cli.py batch -d ./audio_files -r -o results.csv -f csv
```

**Filter results:**
```bash
# Only show/save flagged files
./vfw_cli.py batch -d ./data --filter flagged -o flagged.csv

# Only show safe files
./vfw_cli.py batch -d ./data --filter safe -o safe.csv

# Only show files with hate detected
./vfw_cli.py batch -d ./data --filter hate -o hate_detected.csv
```

**Options:**
- `-d, --directory DIR` - Directory containing audio files
- `-r, --recursive` - Scan directory recursively
- `-v, --verbose` - Show detailed results for each file
- `-o, --output FILE` - Save results to file
- `-f, --format {json,csv}` - Output format (default: json)
- `--filter {flagged,safe,hate}` - Filter results
- `--continue-on-error` - Continue processing even if a file fails

### `screen` - Screen Training Data (Research Mode)

**The main command for researchers** - comprehensively screens a training data directory and generates a detailed research report.

```bash
# Screen training data directory
./vfw_cli.py screen ./training_data/

# Continue even if some files fail
./vfw_cli.py screen ./training_data/ --continue-on-error
```

**What it does:**
1. Recursively scans directory for all audio files
2. Analyzes every file
3. Generates comprehensive statistics
4. Creates multiple output files:
   - `screening_report_TIMESTAMP.json` - Full detailed results
   - `screening_report_TIMESTAMP.csv` - Segment-level data for analysis
   - `flagged_files_TIMESTAMP.txt` - List of problematic files (if any)

**Output includes:**
- Overall dataset statistics
- Files categorized by safety level
- Segment-level analysis
- Processing time metrics
- Recommendations for dataset use

**Example output:**
```
======================================================================
SCREENING REPORT - Training Data Analysis
======================================================================
Directory: ./training_data
Generated: 2025-10-05 14:23:45
Processing time: 123.4s (1.23s per file)

======================================================================
FILES ANALYZED: 100
  ✓ Safe: 92 (92.0%)
  🚨 Hate detected: 5 (5.0%)
  ⚠️  Uncertain: 3 (3.0%)

SEGMENT ANALYSIS:
  Total segments: 1,245
  Flagged segments: 47 (3.8%)

RECOMMENDATION:
  ⚠️  Minor issues detected - review 5 flagged files
======================================================================
```

## 📊 Output Formats

### JSON Format

Complete structured data, ideal for further processing:

```json
{
  "audio_path": "file.wav",
  "transcript": "Full transcript...",
  "overall_label": "hate_detected",
  "confidence": 0.87,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "segment text",
      "label": "hate",
      "confidence": 0.92
    }
  ],
  "flagged_count": 2,
  "total_segments": 10
}
```

### CSV Format

Flattened segment-level data, perfect for spreadsheet analysis:

| audio_file | overall_label | segment_start | segment_end | segment_text | segment_label | segment_confidence |
|------------|---------------|---------------|-------------|--------------|---------------|--------------------|
| file.wav   | hate_detected | 0.0           | 3.5         | text here    | hate          | 0.92               |

Each row represents one segment, making it easy to:
- Sort by confidence scores
- Filter specific labels
- Calculate statistics
- Create visualizations


## 📈 Typical Research Workflow

```bash
# 1. Obtain new training data
cd ~/datasets/new_speech_corpus

# 2. Screen the data
~/vocal-firewall/vfw screen .

# 3. Review results
cat screening_report_*.txt

# 4. Export for detailed analysis
~/vocal-firewall/vfw batch -d . -r -o full_analysis.csv -f csv

# 5. Analyze in Python/R
python analyze_results.py full_analysis.csv

# 6. Remove flagged files if needed
# (based on manual review)

# 7. Verify cleaned dataset
~/vocal-firewall/vfw screen .

# 8. Proceed with training
train_model.py --data .
```


## 🎯 Research Use Cases

### 1. Screen Training Data Before Model Training

```bash
# Screen entire dataset
./vfw_cli.py screen ./speech_training_data/

# Review flagged files
cat ./speech_training_data/flagged_files_*.txt

# Remove problematic files or create clean subset
```

### 2. Batch Analysis with Filtering

```bash
# Extract only problematic samples for review
./vfw_cli.py batch -d ./samples -r --filter flagged -o review_needed.csv

# Verify cleaned dataset is safe
./vfw_cli.py batch -d ./cleaned_samples -r --filter hate -o should_be_empty.csv
```

### 3. Dataset Quality Assessment

```bash
# Analyze multiple datasets and compare
./vfw_cli.py screen ./dataset_v1/ > report_v1.txt
./vfw_cli.py screen ./dataset_v2/ > report_v2.txt

# Generate CSV for statistical analysis
./vfw_cli.py batch -d ./dataset -r -o analysis.csv -f csv
# Import analysis.csv into R/Python for detailed statistics
```

### 4. Continuous Integration Testing

```bash
#!/bin/bash
# CI script to verify dataset quality

./vfw_cli.py screen ./new_training_data/

# Check if any hate speech detected
if grep -q "Hate detected: 0" screening_report_*.txt; then
    echo "✓ Dataset is clean"
    exit 0
else
    echo "✗ Dataset contains problematic content"
    exit 1
fi
```

## 🔗 API Configuration

By default, the CLI connects to `http://localhost:8000`.

To use a different API server:

```bash
# Environment variable (persistent)
export VFW_API_URL=http://192.168.1.100:8000
./vfw_cli.py analyze audio.wav

# Command-line flag (one-time)
./vfw_cli.py --api-url http://remote-server:8000 analyze audio.wav
```


## 📁 Supported Audio Formats

- `.wav` - WAVE audio
- `.mp3` - MPEG audio
- `.ogg` - Ogg Vorbis
- `.flac` - Free Lossless Audio Codec
- `.m4a` - MPEG-4 Audio
- `.webm` - WebM audio


## 💡 Tips for Researchers

1. **Use CSV format for analysis**: Easier to import into R, Python pandas, Excel
   ```bash
   ./vfw_cli.py screen ./data/
   # Import screening_report_*.csv into your analysis tools
   ```

2. **Filter before saving**: Reduce file sizes by filtering
   ```bash
   ./vfw_cli.py batch -d ./data -r --filter flagged -o flagged_only.csv
   ```

3. **Process in stages**: For large datasets, process subdirectories separately
   ```bash
   for dir in ./data/*/; do
       ./vfw_cli.py screen "$dir"
   done
   ```

4. **Automate with scripts**: Integrate into your data pipeline
   ```python
   import subprocess
   import json
   
   result = subprocess.run([
       './vfw_cli.py', 'analyze', 'audio.wav', 
       '-o', 'result.json', '-f', 'json'
   ])
   
   with open('result.json') as f:
       data = json.load(f)
       if data[0]['flagged_count'] > 0:
           # Handle flagged content
           pass
   ```


## 🐛 Troubleshooting

**"Failed to connect to API"**
- Ensure API server is running: `./run_api.sh`
- Check the API URL: `./vfw_cli.py health`
- Verify firewall settings if using remote server

**"No audio files found"**
- Check file extensions (must be .wav, .mp3, etc.)
- Use `-r` flag for recursive directory scanning
- Verify directory path is correct

**"Analysis failed"**
- Check audio file is not corrupted
- Ensure file format is supported
- Use `--continue-on-error` flag for batch processing


## 📖 Additional Resources

- **Practical Examples**: See [CLI_EXAMPLES.md](CLI_EXAMPLES.md) for real-world research workflows
- **API Documentation**: http://localhost:8000/docs
- **Main Project README**: [README.md](README.md)
- **Quick Help**: Run `./vfw --help` or `./vfw <command> --help`

---

**Happy researching! 🔬**
