# src/model/run_batch.py
"""
Batch runner for YouTube videos.
- Preferred: calls FastAPI endpoint POST /analyze_url with {"url": "<youtube url>"}.
- Fallback: runs locally by downloading audio with yt-dlp and using UnifiedAnalyzer.

Outputs:
- out/<ytid>.json for each video
- out/report.html with a simple color-coded summary

Usage:
  python -m src.model.run_batch \
    --api http://localhost:8000 \
    --urls https://www.youtube.com/watch?v=ZMhyG27O480 \
           https://www.youtube.com/watch?v=xCnRJvD6kIw \
           https://www.youtube.com/watch?v=v8SargFU548 \
           https://www.youtube.com/watch?v=1lxmsPbyFio

You can omit --api; the script will auto-fallback to local mode.
Requires: yt-dlp installed and available on PATH for local mode.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Optional: requests only if API is provided
try:
    import requests  # type: ignore
except Exception:
    requests = None  # gracefully handle if not installed

# If we run locally, import the analyzer
try:
    # Use debug analyzer for fast testing
    from .debug_analyzer import DebugAnalyzer as UnifiedAnalyzer
    print("📋 Using Debug Analyzer (fast mode)")
except Exception:
    try:
        # Fallback to real analyzer
        from src.pipeline.analyzer import UnifiedAnalyzer
        print("🔬 Using Real Analyzer (slow mode)")
    except Exception:
        UnifiedAnalyzer = None  # will only be used when needed


DEFAULT_URLS = [
    "https://www.youtube.com/watch?v=ZMhyG27O480",
    "https://www.youtube.com/watch?v=xCnRJvD6kIw",
    "https://www.youtube.com/watch?v=v8SargFU548",
    "https://www.youtube.com/watch?v=1lxmsPbyFio",
]


def extract_ytid(url: str) -> str:
    m = re.search(r"(?:v=|be/)([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[:16]


def call_api(api_base: str, url: str) -> Optional[Dict[str, Any]]:
    if not api_base or not requests:
        return None
    endpoint = api_base.rstrip("/") + "/analyze_url"
    try:
        r = requests.post(endpoint, json={"url": url}, timeout=600)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[API] {url} -> HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"[API] failed ({e}); falling back to local.", file=sys.stderr)
        return None


def run_local(url: str) -> Dict[str, Any]:
    if UnifiedAnalyzer is None:
        raise RuntimeError("UnifiedAnalyzer import failed; cannot run local fallback.")
    
    analyzer = UnifiedAnalyzer()
    
    # Check if this is the debug analyzer (fast mode)
    if hasattr(analyzer, '__class__') and 'Debug' in analyzer.__class__.__name__:
        print("🏃‍♂️ Fast debug mode - skipping actual download")
        # Just use the URL as a mock audio path
        return analyzer.analyze(url)
    else:
        print("⏳ Full mode - downloading audio...")
        # Download bestaudio as wav
        with tempfile.TemporaryDirectory() as tmpd:
            templ = str(Path(tmpd) / "%(id)s.%(ext)s")
            cmd = [
                "yt-dlp",
                "-x",
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",
                "-o",
                templ,
                url,
            ]
            subprocess.run(cmd, check=True)
            wavs = list(Path(tmpd).glob("*.wav"))
            if not wavs:
                raise RuntimeError("yt-dlp did not produce a WAV file.")
            return analyzer.analyze(str(wavs[0]))


def save_json(obj: Dict[str, Any], out_dir: Path, ytid: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ytid}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path


def _cell_color(p: float) -> str:
    """
    Turn prob into a background color (green for low, red for high).
    Uses hue: 120 (green) -> 0 (red).
    """
    p = max(0.0, min(1.0, p))
    hue = 120 * (1.0 - p)  # 0=red, 120=green
    return f"background: hsl({hue:.0f}, 70%, 80%);"


def write_report(results: Dict[str, Dict[str, Any]], out_dir: Path) -> Path:
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Extremism Batch Report</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;padding:20px;} "
        "table{border-collapse:collapse;margin:16px 0;width:100%;} "
        "th,td{border:1px solid #ddd;padding:8px;text-align:left;} "
        "th{background:#f7f7f7;} .pill{display:inline-block;padding:2px 8px;border-radius:12px;font-weight:600;} "
        ".ext{background:#ffebee;color:#b71c1c;} .pot{background:#fff8e1;color:#ef6c00;} .non{background:#e8f5e9;color:#1b5e20;}"
        "</style></head><body>",
        "<h1>Extremism Batch Report</h1>",
    ]

    def pill(label: str) -> str:
        cls = "non"
        if label == "extremist":
            cls = "ext"
        elif label == "potentially_extremist":
            cls = "pot"
        return f"<span class='pill {cls}'>{label}</span>"

    for ytid, data in results.items():
        final = data.get("final", {})
        label = final.get("label", "unknown")
        conf = final.get("confidence", 0.0)
        html += [
            f"<h2>{ytid} &nbsp; {pill(label)} &nbsp; <small>confidence: {conf:.2f}</small></h2>",
            "<table>",
            "<tr><th>#</th><th>Start–End</th><th>Text</th>"
            "<th>p(extremist)</th><th>p(potential)</th><th>p(non)</th></tr>",
        ]

        # Sort utterances by p(extremist) desc; show top 12 for brevity
        utts = data.get("utterances", [])
        def pext(u): return (u.get("probs") or [0,0,0])[0]
        utts_sorted = sorted(utts, key=pext, reverse=True)[:12]

        for i, u in enumerate(utts_sorted, 1):
            p = u.get("probs", [0.0, 0.0, 0.0])
            start = u.get("start", 0.0)
            end = u.get("end", 0.0)
            text = (u.get("text") or "").replace("<", "&lt;").replace(">", "&gt;")
            html += [
                "<tr>",
                f"<td>{i}</td>",
                f"<td>{start:.1f}s–{end:.1f}s</td>",
                f"<td>{text}</td>",
                f"<td style='{_cell_color(p[0])}'>{p[0]:.2f}</td>",
                f"<td style='{_cell_color(p[1])}'>{p[1]:.2f}</td>",
                f"<td style='{_cell_color(p[2])}'>{p[2]:.2f}</td>",
                "</tr>",
            ]
        html.append("</table>")

    html.append("</body></html>")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", type=str, default="http://localhost:8000",
                   help="Base URL of your FastAPI server (where /analyze_url lives). "
                        "If unreachable or returns error, the script falls back to local mode.")
    p.add_argument("--urls", nargs="*", default=DEFAULT_URLS, help="YouTube URLs")
    p.add_argument("--out", type=str, default="out", help="Output directory for JSON and report")
    args = p.parse_args()

    out_dir = Path(args.out)
    results: Dict[str, Dict[str, Any]] = {}

    for url in args.urls:
        ytid = extract_ytid(url)
        print(f"=== Processing {url} [{ytid}] ===")
        data = call_api(args.api, url)
        if data is None:
            print("-> Using local fallback…")
            data = run_local(url)
        save_json(data, out_dir, ytid)
        results[ytid] = data
        print(f"Saved: {out_dir / (ytid + '.json')}")

    report_path = write_report(results, out_dir)
    print(f"\nReport: {report_path.resolve()}")


if __name__ == "__main__":
    main()
