import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np
import matplotlib.pyplot as plt

# Execute the existing test module and reuse its trial function.
# This also regenerates the original robustness figures.
import test_robustness

Path("results").mkdir(exist_ok=True)

q_scales = [1.0, 0.1, 0.01]
results = []

for scale in q_scales:
    print(f"\nTesting Q scale: {scale}")

    result = test_robustness.run_trial(
        noise_std=0.0005,
        seed=43,
        q_scale=scale
    )

    results.append((scale, result))

    print(
        f"Status: {result['status']}\n"
        f"RMSE: {result['localisation_rmse_m']:.4f} m\n"
        f"Maximum error: {result['max_localisation_error_m']:.4f} m\n"
        f"Final uncertainty: {result['final_uncertainty_m']:.4f} m\n"
        f"Docking distance: {result['dock_distance_m']:.4f} m"
    )

with open(
    "results/ekf_tuning_results.csv",
    "w",
    newline=""
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "q_scale",
        "status",
        "rmse_m",
        "max_error_m",
        "final_uncertainty_m",
        "dock_distance_m"
    ])

    for scale, r in results:
        writer.writerow([
            scale,
            r["status"],
            r["localisation_rmse_m"],
            r["max_localisation_error_m"],
            r["final_uncertainty_m"],
            r["dock_distance_m"]
        ])

fig, ax = plt.subplots(figsize=(9, 5))

labels = [f"{scale:g}x Q" for scale, _ in results]
rmse = [
    r["localisation_rmse_m"]
    for _, r in results
]
uncertainty = [
    r["final_uncertainty_m"]
    for _, r in results
]

x = np.arange(len(labels))
width = 0.35

ax.bar(
    x - width/2, rmse,
    width, label="Position RMSE"
)
ax.bar(
    x + width/2, uncertainty,
    width, label="Final estimated uncertainty"
)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Metres")
ax.set_title("EKF Process Noise Tuning Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    "results/ekf_tuning_comparison.png",
    dpi=180
)
plt.close(fig)

print("\nEKF tuning comparison complete.")
