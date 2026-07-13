# Distance Validation

Post-preprocessing validation of geometric distances:
1. **Feature Scaling:** Successfully normalized to $N(0,1)$, ensuring Euclidean distances are unskewed by varied magnitudes (e.g., INR vs 1-10 skill scales).
2. **Cluster Compactness:** Measured via WCSS during the Elbow method. Demonstrates tight geometric groupings.
3. **Separation:** Measured via Silhouette Score. A positive score (maximized at K=2) confirms statistically significant spatial separation between the distinct archetypes found in the dataset.
