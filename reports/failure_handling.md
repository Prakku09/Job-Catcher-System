# Failure Handling Guide (Clustering Preprocessing)

- **Missing Dataset:** Pipeline gracefully aborts with `FileNotFoundError` if `clean_modelling_table.csv` is missing.
- **Invalid Numeric Values/Missing Features:** Fallback to safe 0.0 filling or median imputation to ensure distance math does not crash.
- **PCA Failure:** Configured to dynamically adapt to matrix rank. If highly correlated, PCA retains fewest components automatically.
- **Scaling Failure:** Ensure zero-variance columns are flagged in validation, which prevents ZeroDivision errors during `StandardScaler` transformations.
