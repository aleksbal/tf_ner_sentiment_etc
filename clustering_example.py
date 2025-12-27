"""Example: clustering similar objects using simple k-means.

Use case: segment retail customers based on their purchase behavior.
Each customer has two features:
- average monthly spend (USD)
- number of monthly purchases
"""

from __future__ import annotations

import numpy as np


def kmeans(points: np.ndarray, k: int, iterations: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Cluster points with a minimal k-means implementation.

    Returns:
        centers: (k, n_features) cluster centers
        labels: (n_points,) cluster assignment for each point
    """
    rng = np.random.default_rng(42)
    centers = points[rng.choice(len(points), size=k, replace=False)]

    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array([
            points[labels == idx].mean(axis=0) if np.any(labels == idx) else centers[idx]
            for idx in range(k)
        ])

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return centers, labels


if __name__ == "__main__":
    customers = np.array([
        [35, 2],   # budget buyer
        [40, 3],
        [45, 2],
        [120, 4],  # steady mid-tier
        [135, 5],
        [150, 6],
        [300, 8],  # high value
        [320, 9],
        [340, 8],
    ])

    centers, labels = kmeans(customers, k=3, iterations=20)

    print("Cluster centers (avg spend, purchases):")
    for idx, center in enumerate(centers, start=1):
        print(f"  Cluster {idx}: spend=${center[0]:.1f}, purchases={center[1]:.1f}")

    print("\nCustomer assignments:")
    for customer, label in zip(customers, labels, strict=False):
        print(f"  Customer {customer.tolist()} -> Cluster {label + 1}")
