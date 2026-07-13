# Trade-off Analysis

| Method | Benefit | Limitation |
|---|---|---|
| **StandardScaler** | Equal feature geometric contribution | Highly sensitive to extreme outliers |
| **PCA** | Faster clustering, removes noise/collinearity | Lower interpretability of axis meanings |
| **Elbow (WCSS)** | Very simple geometric intuition | Highly subjective to human interpretation |
| **Silhouette** | High quality, objective cohesion estimate | Geometrically higher computational cost $O(N^2)$ |
