# 📚 Vocal Firewall CLI - Practical Examples

Real-world examples for researchers using the CLI tool.

## 🎓 Example 1: Screen New Training Dataset

**Scenario**: You've downloaded a new speech dataset and want to check if it contains extremist content before using it for training.

```bash
# 1. Start the API server (if not already running)
./run_api.sh

# 2. Screen the entire dataset
./vfw screen ./datasets/common_voice_subset/

# Output will show:
# - Total files analyzed
# - Files categorized by safety level
# - Segment-level statistics
# - Recommendations
```

**Generated files:**
- `screening_report_TIMESTAMP.json` - Full results for programmatic access
- `screening_report_TIMESTAMP.csv` - Segment-level data for analysis
- `flagged_files_TIMESTAMP.txt` - List of problematic files (if any)

**What to do next:**
- If 0-5% flagged: Review flagged files manually, decide if removable
- If 5-15% flagged: Carefully review, may need significant cleaning
- If >15% flagged: Consider alternative dataset or extensive curation

---

## 📊 Example 2: Generate Analysis Dataset for Paper

**Scenario**: You're writing a research paper and need detailed statistics about extremist content in various datasets.

```bash
# Analyze multiple datasets
./vfw screen ./datasets/librispeech_subset/ > librispeech_report.txt
./vfw screen ./datasets/common_voice_subset/ > common_voice_report.txt
./vfw screen ./datasets/voxceleb_subset/ > voxceleb_report.txt

# Generate CSV files for statistical analysis
./vfw batch -d ./datasets/librispeech_subset -r -o librispeech_analysis.csv -f csv
./vfw batch -d ./datasets/common_voice_subset -r -o common_voice_analysis.csv -f csv
./vfw batch -d ./datasets/voxceleb_subset -r -o voxceleb_analysis.csv -f csv
```

**Analysis in Python:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
datasets = {
    'LibriSpeech': pd.read_csv('librispeech_analysis.csv'),
    'Common Voice': pd.read_csv('common_voice_analysis.csv'),
    'VoxCeleb': pd.read_csv('voxceleb_analysis.csv')
}

# Compare flagged content rates
for name, df in datasets.items():
    total_files = df['audio_file'].nunique()
    flagged_files = df[df['flagged_count'] > 0]['audio_file'].nunique()
    rate = (flagged_files / total_files) * 100
    print(f"{name}: {rate:.2f}% files flagged")

# Analyze by confidence levels
for name, df in datasets.items():
    high_conf_hate = df[
        (df['segment_label'] == 'hate') & 
        (df['segment_confidence'] > 0.8)
    ]
    print(f"{name}: {len(high_conf_hate)} high-confidence hate segments")
```

---

## 🧹 Example 3: Clean Dataset - Remove Problematic Files

**Scenario**: Create a cleaned version of your dataset by removing flagged files.

```bash
# 1. Screen and get flagged files list
./vfw screen ./raw_dataset/

# 2. Extract just the filenames from the flagged files list
grep "\.wav\|\.mp3\|\.flac" flagged_files_*.txt | \
    awk '{print $1}' > files_to_remove.txt

# 3. Create cleaned dataset (copy good files only)
mkdir -p ./cleaned_dataset

# 4. Process in Python for safety
python3 << EOF
import shutil
from pathlib import Path

# Read flagged files
with open('files_to_remove.txt') as f:
    flagged = {line.strip() for line in f}

# Copy non-flagged files
raw_dir = Path('./raw_dataset')
clean_dir = Path('./cleaned_dataset')

for audio_file in raw_dir.rglob('*.wav'):
    if str(audio_file) not in flagged:
        rel_path = audio_file.relative_to(raw_dir)
        dest = clean_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_file, dest)
        print(f"Copied: {rel_path}")

print(f"\nCleaned dataset created in {clean_dir}")
EOF

# 5. Verify the cleaned dataset
./vfw screen ./cleaned_dataset/
```

---

## 🔍 Example 4: Focus on High-Risk Content

**Scenario**: You only want to review files with confirmed hate speech for removal.

```bash
# Get only files with hate detected (not just uncertain)
./vfw batch -d ./dataset -r --filter hate -o high_risk.csv -f csv

# Review the CSV
column -s, -t high_risk.csv | less -S

# Export just the audio file paths
cut -d',' -f1 high_risk.csv | sort -u > confirmed_hate_files.txt

# Count
wc -l confirmed_hate_files.txt
```

---

## 🤖 Example 5: Integrate into Training Pipeline

**Scenario**: Automatically screen new data before adding to training set.

**screening_pipeline.sh:**
```bash
#!/bin/bash
set -e

NEW_DATA_DIR="$1"
CLEAN_DATA_DIR="./training_data/clean"
QUARANTINE_DIR="./training_data/quarantine"

echo "🔍 Screening new data: $NEW_DATA_DIR"

# Run screening
./vfw screen "$NEW_DATA_DIR" > screening_log.txt

# Check results
HATE_COUNT=$(grep "Hate detected:" screening_log.txt | awk '{print $3}')

if [ "$HATE_COUNT" -eq 0 ]; then
    echo "✓ All clean! Moving to training data..."
    rsync -av "$NEW_DATA_DIR/" "$CLEAN_DATA_DIR/"
    echo "✓ Data added to training set"
else
    echo "⚠️  Found $HATE_COUNT problematic files"
    echo "Moving to quarantine for manual review..."
    
    # Move flagged files to quarantine
    mkdir -p "$QUARANTINE_DIR"
    
    # Extract and move flagged files
    grep "\.wav\|\.mp3" flagged_files_*.txt | while read -r file; do
        if [ -f "$file" ]; then
            mv "$file" "$QUARANTINE_DIR/"
            echo "Quarantined: $(basename $file)"
        fi
    done
    
    # Move clean files to training data
    rsync -av "$NEW_DATA_DIR/" "$CLEAN_DATA_DIR/"
    
    echo "⚠️  Manual review required for quarantined files"
    exit 1
fi
```

**Usage:**
```bash
chmod +x screening_pipeline.sh
./screening_pipeline.sh ./incoming_data/batch_001/
```

---

## 📈 Example 6: Monitor Dataset Quality Over Time

**Scenario**: Track quality metrics as you add more data to your dataset.

**monitor_quality.sh:**
```bash
#!/bin/bash

DATASET_DIR="./training_data"
REPORTS_DIR="./quality_reports"
mkdir -p "$REPORTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORTS_DIR/quality_report_$TIMESTAMP.txt"

# Run screening
./vfw screen "$DATASET_DIR" > "$REPORT_FILE"

# Extract key metrics
TOTAL=$(grep "FILES ANALYZED:" "$REPORT_FILE" | awk '{print $3}')
SAFE=$(grep "Safe:" "$REPORT_FILE" | grep -o '[0-9]*' | head -1)
HATE=$(grep "Hate detected:" "$REPORT_FILE" | grep -o '[0-9]*' | head -1)
SAFE_PCT=$(grep "Safe:" "$REPORT_FILE" | grep -o '[0-9.]*%' | head -1)

# Log to CSV for tracking
echo "$TIMESTAMP,$TOTAL,$SAFE,$HATE,$SAFE_PCT" >> "$REPORTS_DIR/quality_timeline.csv"

# Show trend
echo "Quality Trend:"
tail -10 "$REPORTS_DIR/quality_timeline.csv" | column -s, -t
```

**Track over time:**
```bash
# Initial dataset
./monitor_quality.sh

# After adding batch 1
rsync -av ./new_batch_1/ ./training_data/
./monitor_quality.sh

# After adding batch 2
rsync -av ./new_batch_2/ ./training_data/
./monitor_quality.sh

# View trend
cat ./quality_reports/quality_timeline.csv
```

---

## 🔬 Example 7: Compare Pre/Post Cleaning

**Scenario**: Measure the effectiveness of your cleaning process.

```bash
# Before cleaning
echo "=== Before Cleaning ===" > comparison.txt
./vfw screen ./dataset_raw >> comparison.txt

# Clean the dataset (your process here)
# ...

# After cleaning
echo -e "\n\n=== After Cleaning ===" >> comparison.txt
./vfw screen ./dataset_cleaned >> comparison.txt

# View comparison
cat comparison.txt

# Generate detailed CSVs for analysis
./vfw batch -d ./dataset_raw -r -o before_cleaning.csv -f csv
./vfw batch -d ./dataset_cleaned -r -o after_cleaning.csv -f csv
```

**Analysis in Python:**
```python
import pandas as pd

before = pd.read_csv('before_cleaning.csv')
after = pd.read_csv('after_cleaning.csv')

print("Dataset Cleaning Impact:")
print(f"Files before: {before['audio_file'].nunique()}")
print(f"Files after: {after['audio_file'].nunique()}")
print(f"Files removed: {before['audio_file'].nunique() - after['audio_file'].nunique()}")
print(f"\nFlagged segments before: {before['flagged_count'].sum()}")
print(f"Flagged segments after: {after['flagged_count'].sum()}")
print(f"Reduction: {((before['flagged_count'].sum() - after['flagged_count'].sum()) / before['flagged_count'].sum() * 100):.1f}%")
```

---

## 🌐 Example 8: Remote API Server

**Scenario**: Your API server is on a different machine (e.g., GPU server).

```bash
# Set API URL once
export VFW_API_URL="http://192.168.1.100:8000"

# Or use flag each time
./vfw --api-url http://192.168.1.100:8000 screen ./local_dataset/

# Check remote server health
./vfw --api-url http://gpu-server:8000 health
```

---

## 🔄 Example 9: Continuous Integration Check

**Scenario**: Automatically verify dataset quality in CI/CD pipeline.

**.github/workflows/dataset_quality.yml:**
```yaml
name: Dataset Quality Check

on:
  push:
    paths:
      - 'training_data/**'

jobs:
  quality-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Start API Server
      run: |
        ./run_api.sh &
        sleep 30  # Wait for models to load
    
    - name: Screen Dataset
      run: |
        ./vfw screen ./training_data/ > screening_results.txt
    
    - name: Check for hate speech
      run: |
        HATE_COUNT=$(grep "Hate detected:" screening_results.txt | grep -o '[0-9]*' | head -1)
        if [ "$HATE_COUNT" -gt 0 ]; then
          echo "❌ Dataset contains $HATE_COUNT files with hate speech"
          exit 1
        else
          echo "✓ Dataset quality check passed"
        fi
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: screening-results
        path: |
          screening_results.txt
          screening_report_*.json
          screening_report_*.csv
```

---

## 💾 Example 10: Export for External Analysis

**Scenario**: Export data in various formats for different analysis tools.

```bash
# For Python/Pandas analysis
./vfw batch -d ./dataset -r -o analysis.csv -f csv

# For JSON-based tools
./vfw batch -d ./dataset -r -o analysis.json -f json

# For Excel/Spreadsheet (CSV works)
./vfw batch -d ./dataset -r -o analysis.csv -f csv

# For R analysis
./vfw batch -d ./dataset -r -o analysis.csv -f csv
# Then in R: data <- read.csv("analysis.csv")

# Extract just summary statistics
./vfw screen ./dataset > summary.txt
grep -E "FILES ANALYZED|Safe:|Hate detected:|SEGMENT ANALYSIS" summary.txt > quick_summary.txt
```

---

## 🎯 Tips for Large Datasets

```bash
# 1. Process in subdirectories to track progress
for subdir in ./large_dataset/*/; do
    echo "Processing: $subdir"
    ./vfw screen "$subdir"
done

# 2. Use continue-on-error for resilience
./vfw batch -d ./large_dataset -r --continue-on-error -o results.csv

# 3. Process incrementally and merge results
./vfw batch -d ./large_dataset/part1 -r -o part1.csv
./vfw batch -d ./large_dataset/part2 -r -o part2.csv
./vfw batch -d ./large_dataset/part3 -r -o part3.csv

# Merge CSVs (keeping only one header)
head -1 part1.csv > merged.csv
tail -n +2 part1.csv >> merged.csv
tail -n +2 part2.csv >> merged.csv
tail -n +2 part3.csv >> merged.csv
```

---

## 📖 Additional Resources

- Full CLI documentation: `CLI_README.md`
- API documentation: http://localhost:8000/docs
- Main project README: `README.md`

**Need help?** Run `./vfw --help` or `./vfw <command> --help`
