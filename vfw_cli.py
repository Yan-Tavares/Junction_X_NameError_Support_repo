#!/usr/bin/env python3
"""
Vocal Firewall CLI
Command-line interface for extremist speech detection API

Designed for research purposes, especially for screening training data for speech models.
"""

import argparse
import json
import sys
import csv
from pathlib import Path
from typing import List, Dict, Optional
import requests
from datetime import datetime
import time


class VocalFirewallCLI:
    """CLI client for Vocal Firewall API"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
    
    def check_health(self) -> Dict:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to API: {e}")
    
    def analyze_file(self, audio_path: Path) -> Dict:
        """Analyze a single audio file"""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path.name, f, self._get_mime_type(audio_path))}
            response = self.session.post(f"{self.api_url}/analyze", files=files)
            response.raise_for_status()
            return response.json()
    
    def analyze_batch(self, audio_files: List[Path]) -> Dict:
        """Analyze multiple audio files in batch"""
        files = []
        for audio_path in audio_files:
            if audio_path.exists():
                files.append(('files', (audio_path.name, open(audio_path, 'rb'), self._get_mime_type(audio_path))))
        
        try:
            response = self.session.post(f"{self.api_url}/analyze/batch", files=files)
            response.raise_for_status()
            return response.json()
        finally:
            # Close all file handles
            for _, (_, f, _) in files:
                f.close()
    
    @staticmethod
    def _get_mime_type(audio_path: Path) -> str:
        """Get MIME type based on file extension"""
        ext_map = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.webm': 'audio/webm'
        }
        return ext_map.get(audio_path.suffix.lower(), 'application/octet-stream')


def format_result_text(result: Dict, verbose: bool = False) -> str:
    """Format analysis result as human-readable text"""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"File: {result['audio_path']}")
    lines.append(f"Overall Label: {result['overall_label'].upper()}")
    lines.append(f"Confidence: {result['confidence']:.2%}")
    lines.append(f"Flagged Segments: {result['flagged_count']}/{result['total_segments']}")
    lines.append(f"{'='*70}")
    
    if verbose:
        lines.append(f"\nTranscript:\n{result['transcript']}\n")
        
        if result['segments']:
            lines.append(f"\nSegment Analysis:")
            lines.append(f"{'-'*70}")
            for seg in result['segments']:
                label_marker = "🚨" if seg['label'] == 'hate' else "⚠️" if seg['label'] == 'uncertain' else "✓"
                lines.append(f"{label_marker} [{seg['start']:.2f}s - {seg['end']:.2f}s] "
                           f"{seg['label'].upper()} (conf: {seg['confidence']:.2%})")
                lines.append(f"   \"{seg['text']}\"")
            lines.append(f"{'-'*70}")
    
    return "\n".join(lines)


def save_results_json(results: List[Dict], output_path: Path):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_path}")


def save_results_csv(results: List[Dict], output_path: Path):
    """Save results to CSV file (flattened for research use)"""
    rows = []
    for result in results:
        # One row per segment for detailed analysis
        for seg in result['segments']:
            rows.append({
                'audio_file': result['audio_path'],
                'overall_label': result['overall_label'],
                'overall_confidence': result['confidence'],
                'segment_start': seg['start'],
                'segment_end': seg['end'],
                'segment_text': seg['text'],
                'segment_label': seg['label'],
                'segment_confidence': seg['confidence'],
                'flagged_count': result['flagged_count'],
                'total_segments': result['total_segments']
            })
    
    if rows:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Results saved to {output_path}")


def filter_results(results: List[Dict], filter_type: Optional[str]) -> List[Dict]:
    """Filter results based on criteria"""
    if not filter_type:
        return results
    
    if filter_type == 'flagged':
        return [r for r in results if r['flagged_count'] > 0]
    elif filter_type == 'safe':
        return [r for r in results if r['overall_label'] == 'safe']
    elif filter_type == 'hate':
        return [r for r in results if r['overall_label'] == 'hate_detected']
    else:
        return results


def scan_directory(directory: Path, recursive: bool = False) -> List[Path]:
    """Scan directory for audio files"""
    audio_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'}
    
    if recursive:
        files = [f for f in directory.rglob('*') if f.suffix.lower() in audio_extensions]
    else:
        files = [f for f in directory.glob('*') if f.suffix.lower() in audio_extensions]
    
    return sorted(files)


def cmd_analyze(args):
    """Analyze single audio file"""
    cli = VocalFirewallCLI(args.api_url)
    
    # Check API health
    try:
        health = cli.check_health()
        if not health.get('models_loaded'):
            print("⚠️  Warning: API models not fully loaded", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error: Could not connect to API at {args.api_url}", file=sys.stderr)
        print(f"   {e}", file=sys.stderr)
        return 1
    
    # Analyze file
    print(f"🔍 Analyzing {args.audio_file}...")
    try:
        result = cli.analyze_file(Path(args.audio_file))
        
        # Display result
        print(format_result_text(result, verbose=args.verbose))
        
        # Save if requested
        if args.output:
            output_path = Path(args.output)
            if args.format == 'json':
                save_results_json([result], output_path)
            elif args.format == 'csv':
                save_results_csv([result], output_path)
        
        return 0
    except Exception as e:
        print(f"❌ Analysis failed: {e}", file=sys.stderr)
        return 1


def cmd_batch(args):
    """Batch analyze multiple files"""
    cli = VocalFirewallCLI(args.api_url)
    
    # Check API health
    try:
        health = cli.check_health()
        if not health.get('models_loaded'):
            print("⚠️  Warning: API models not fully loaded", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error: Could not connect to API at {args.api_url}", file=sys.stderr)
        print(f"   {e}", file=sys.stderr)
        return 1
    
    # Gather files
    if args.directory:
        audio_files = scan_directory(Path(args.directory), recursive=args.recursive)
        if not audio_files:
            print(f"❌ No audio files found in {args.directory}", file=sys.stderr)
            return 1
        print(f"📁 Found {len(audio_files)} audio files")
    else:
        audio_files = [Path(f) for f in args.files]
    
    # Analyze files
    results = []
    print(f"🔍 Analyzing {len(audio_files)} files...\n")
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing {audio_file.name}...", end=' ')
        try:
            result = cli.analyze_file(audio_file)
            results.append(result)
            
            # Show quick summary
            label_emoji = "🚨" if result['overall_label'] == 'hate_detected' else "⚠️" if result['overall_label'] == 'uncertain' else "✓"
            print(f"{label_emoji} {result['overall_label']} ({result['flagged_count']} flagged)")
        except Exception as e:
            print(f"❌ Failed: {e}")
            if args.continue_on_error:
                continue
            else:
                return 1
    
    # Apply filters
    filtered_results = filter_results(results, args.filter)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Batch Analysis Complete")
    print(f"{'='*70}")
    print(f"Total processed: {len(results)}")
    print(f"Flagged: {sum(1 for r in results if r['flagged_count'] > 0)}")
    print(f"Safe: {sum(1 for r in results if r['overall_label'] == 'safe')}")
    print(f"Hate detected: {sum(1 for r in results if r['overall_label'] == 'hate_detected')}")
    
    if args.filter:
        print(f"\nFiltered results ({args.filter}): {len(filtered_results)}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        results_to_save = filtered_results if args.filter else results
        
        if args.format == 'json':
            save_results_json(results_to_save, output_path)
        elif args.format == 'csv':
            save_results_csv(results_to_save, output_path)
    
    # Show detailed results if verbose
    if args.verbose:
        for result in (filtered_results if args.filter else results):
            print(format_result_text(result, verbose=True))
    
    return 0


def cmd_health(args):
    """Check API health"""
    cli = VocalFirewallCLI(args.api_url)
    
    try:
        health = cli.check_health()
        print(f"API Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Models Loaded: {'✓' if health['models_loaded'] else '✗'}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cmd_screen(args):
    """Screen training data directory and generate research report"""
    cli = VocalFirewallCLI(args.api_url)
    
    # Check API health
    try:
        health = cli.check_health()
        if not health.get('models_loaded'):
            print("⚠️  Warning: API models not fully loaded", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error: Could not connect to API at {args.api_url}", file=sys.stderr)
        return 1
    
    # Scan directory
    print(f"📁 Scanning {args.data_dir} for audio files...")
    audio_files = scan_directory(Path(args.data_dir), recursive=True)
    
    if not audio_files:
        print(f"❌ No audio files found", file=sys.stderr)
        return 1
    
    print(f"   Found {len(audio_files)} audio files\n")
    
    # Analyze all files
    results = []
    start_time = time.time()
    
    for i, audio_file in enumerate(audio_files, 1):
        rel_path = audio_file.relative_to(args.data_dir) if args.data_dir else audio_file.name
        print(f"[{i}/{len(audio_files)}] {rel_path}...", end=' ')
        
        try:
            result = cli.analyze_file(audio_file)
            results.append(result)
            
            label_emoji = "🚨" if result['overall_label'] == 'hate_detected' else "⚠️" if result['overall_label'] == 'uncertain' else "✓"
            print(f"{label_emoji} {result['overall_label']}")
        except Exception as e:
            print(f"❌ Error: {e}")
            if not args.continue_on_error:
                return 1
    
    elapsed = time.time() - start_time
    
    # Generate report
    print(f"\n{'='*70}")
    print(f"SCREENING REPORT - Training Data Analysis")
    print(f"{'='*70}")
    print(f"Directory: {args.data_dir}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing time: {elapsed:.1f}s ({elapsed/len(results):.2f}s per file)")
    print(f"\n{'='*70}")
    
    # Statistics
    total = len(results)
    safe = sum(1 for r in results if r['overall_label'] == 'safe')
    hate = sum(1 for r in results if r['overall_label'] == 'hate_detected')
    uncertain = sum(1 for r in results if r['overall_label'] == 'uncertain')
    total_flagged = sum(r['flagged_count'] for r in results)
    total_segments = sum(r['total_segments'] for r in results)
    
    print(f"FILES ANALYZED: {total}")
    print(f"  ✓ Safe: {safe} ({safe/total*100:.1f}%)")
    print(f"  🚨 Hate detected: {hate} ({hate/total*100:.1f}%)")
    print(f"  ⚠️  Uncertain: {uncertain} ({uncertain/total*100:.1f}%)")
    print(f"\nSEGMENT ANALYSIS:")
    print(f"  Total segments: {total_segments}")
    print(f"  Flagged segments: {total_flagged} ({total_flagged/total_segments*100:.1f}%)")
    print(f"\nRECOMMENDATION:")
    
    if hate == 0:
        print(f"  ✓ Dataset appears clean - suitable for training")
    elif hate / total < 0.05:
        print(f"  ⚠️  Minor issues detected - review {hate} flagged files")
    else:
        print(f"  🚨 Significant issues detected - review recommended")
        print(f"     {hate} files contain hate speech ({hate/total*100:.1f}%)")
    
    print(f"{'='*70}\n")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save all results as JSON
    json_path = Path(args.data_dir) / f"screening_report_{timestamp}.json"
    save_results_json(results, json_path)
    
    # Save CSV with segment details
    csv_path = Path(args.data_dir) / f"screening_report_{timestamp}.csv"
    save_results_csv(results, csv_path)
    
    # Save flagged files list
    if hate > 0 or uncertain > 0:
        flagged_path = Path(args.data_dir) / f"flagged_files_{timestamp}.txt"
        with open(flagged_path, 'w') as f:
            f.write(f"Flagged Files - Generated {datetime.now()}\n")
            f.write(f"{'='*70}\n\n")
            for r in results:
                if r['flagged_count'] > 0:
                    f.write(f"{r['audio_path']}\n")
                    f.write(f"  Label: {r['overall_label']}, Confidence: {r['confidence']:.2%}\n")
                    f.write(f"  Flagged segments: {r['flagged_count']}/{r['total_segments']}\n\n")
        print(f"✓ Flagged files list saved to {flagged_path}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Vocal Firewall CLI - Extremist Speech Detection for Research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  %(prog)s analyze audio.wav
  
  # Analyze with verbose output and save to JSON
  %(prog)s analyze audio.wav -v -o results.json
  
  # Batch analyze multiple files
  %(prog)s batch file1.wav file2.wav file3.wav -o results.csv
  
  # Batch analyze directory
  %(prog)s batch -d ./audio_files -r -o results.json
  
  # Screen training data directory (research mode)
  %(prog)s screen ./training_data/
  
  # Filter only flagged results
  %(prog)s batch -d ./data -f flagged -o flagged_only.csv
        """
    )
    
    parser.add_argument('--api-url', default='http://localhost:8000',
                       help='API base URL (default: http://localhost:8000)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check API health status')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze single audio file')
    analyze_parser.add_argument('audio_file', help='Path to audio file')
    analyze_parser.add_argument('-v', '--verbose', action='store_true',
                               help='Show detailed segment analysis')
    analyze_parser.add_argument('-o', '--output', help='Save results to file')
    analyze_parser.add_argument('-f', '--format', choices=['json', 'csv'],
                               default='json', help='Output format (default: json)')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple audio files')
    batch_group = batch_parser.add_mutually_exclusive_group(required=True)
    batch_group.add_argument('files', nargs='*', help='Audio files to analyze')
    batch_group.add_argument('-d', '--directory', help='Directory containing audio files')
    batch_parser.add_argument('-r', '--recursive', action='store_true',
                            help='Scan directory recursively')
    batch_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Show detailed results')
    batch_parser.add_argument('-o', '--output', help='Save results to file')
    batch_parser.add_argument('-f', '--format', choices=['json', 'csv'],
                            default='json', help='Output format (default: json)')
    batch_parser.add_argument('--filter', choices=['flagged', 'safe', 'hate'],
                            help='Filter results (flagged=any issues, safe=clean, hate=hate detected)')
    batch_parser.add_argument('--continue-on-error', action='store_true',
                            help='Continue processing even if a file fails')
    
    # Screen command (research-focused)
    screen_parser = subparsers.add_parser('screen',
                                         help='Screen training data directory (generates research report)')
    screen_parser.add_argument('data_dir', help='Training data directory to screen')
    screen_parser.add_argument('--continue-on-error', action='store_true',
                              help='Continue processing even if a file fails')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to command handlers
    if args.command == 'health':
        return cmd_health(args)
    elif args.command == 'analyze':
        return cmd_analyze(args)
    elif args.command == 'batch':
        return cmd_batch(args)
    elif args.command == 'screen':
        return cmd_screen(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
