"""
Analyze a saved BlinkRate session — plots fatigue over time.
Usage: python analyze_session.py sessions/session_YYYYMMDD_HHMMSS.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def analyze_session(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = data['samples']
    if not samples:
        print("No samples in session.")
        return

    times = [s['timestamp'] / 60 for s in samples]  # minutes
    fatigue = [s['fatigue'] for s in samples]
    blink_rate = [s['blink_rate'] for s in samples]
    perclos = [s['perclos'] for s in samples]
    eard = [s['eard'] for s in samples]

    # --- Figure ---
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.35)

    # 1. Fatigue over time
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, fatigue, 'r-', linewidth=2, label='Fatigue')
    ax1.axhline(0.4, color='orange', linestyle='--', alpha=0.6, label='Tired threshold')
    ax1.axhline(0.7, color='red', linestyle='--', alpha=0.6, label='Fatigued threshold')
    ax1.fill_between(times, 0, fatigue, alpha=0.2, color='red')
    ax1.set_ylabel('Fatigue Score')
    ax1.set_title(f"BlinkRate Session — {data['start_time']}")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left')

    # 2. Blink rate over time
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(times, blink_rate, 'b-', linewidth=2, label='Blinks/min')
    ax2.axhspan(15, 20, alpha=0.2, color='green', label='Normal range')
    ax2.set_ylabel('Blinks / min')
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left')

    # 3. PERCLOS + EARD
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(times, perclos, 'purple', linewidth=2, label='PERCLOS')
    ax3.plot(times, eard, 'teal', linewidth=2, label='EARD (novel)')
    ax3.axhline(0.15, color='red', linestyle='--', alpha=0.6,
                label='Drowsy threshold (PERCLOS)')
    ax3.set_ylabel('Score')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylim(0, 1)
    ax3.grid(alpha=0.3)
    ax3.legend(loc='upper left')

    plt.savefig(filepath.replace('.json', '_report.png'), dpi=100, bbox_inches='tight')
    print(f"📈 Report saved: {filepath.replace('.json', '_report.png')}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f"📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"Duration:              {data['duration_seconds']:.1f} sec")
    print(f"Total blinks:          {data['total_blinks']}")
    print(f"Average blink rate:    {data['avg_blink_rate']:.1f} / min")
    print(f"Average fatigue:       {data['avg_fatigue']:.2%}")
    print(f"Max fatigue:           {data['max_fatigue']:.2%}")
    print(f"Time in 'TIRED':       {sum(1 for f in fatigue if 0.4 <= f < 0.7) * 2} sec")
    print(f"Time in 'FATIGUED':    {sum(1 for f in fatigue if f >= 0.7) * 2} sec")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_session.py <session.json>")
        print("Example: python analyze_session.py sessions/session_20250115_143022.json")
        sys.exit(1)
    analyze_session(sys.argv[1])