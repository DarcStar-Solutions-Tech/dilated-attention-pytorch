#!/usr/bin/env python3
"""
Monitor GPU memory usage in real-time.
Run this in a separate terminal while running distributed tests.
"""

import subprocess
import time
import os


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def get_gpu_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )

    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(", ")
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used": int(parts[2]),
                "memory_free": int(parts[3]),
                "memory_total": int(parts[4]),
                "utilization": int(parts[5]),
            }
        )
    return gpus


def main():
    print("GPU Memory Monitor - Press Ctrl+C to exit")
    print("=" * 70)

    try:
        while True:
            clear_screen()
            print("GPU Memory Monitor - Press Ctrl+C to exit")
            print("=" * 70)
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            gpus = get_gpu_info()

            for gpu in gpus:
                print(f"\nGPU {gpu['index']}: {gpu['name']}")
                print(
                    f"  Memory: {gpu['memory_used']:,}MB / {gpu['memory_total']:,}MB "
                    f"({gpu['memory_used'] / gpu['memory_total'] * 100:.1f}%)"
                )
                print(f"  Free: {gpu['memory_free']:,}MB")
                print(f"  Utilization: {gpu['utilization']}%")

                # Memory bar
                bar_width = 50
                used_bars = int((gpu["memory_used"] / gpu["memory_total"]) * bar_width)
                bar = "█" * used_bars + "░" * (bar_width - used_bars)
                print(f"  [{bar}]")

            print("\n" + "=" * 70)
            time.sleep(0.5)  # Update every 500ms

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
