# Edge Cases Analysis

- **Duplicate Candidates:** Automatically stripped out via `drop_duplicates()` before scaling to prevent artificial clustering density.
- **Extreme Salary Values:** Minimized via robust `StandardScaler` application. 
- **Constant Features:** `dataset_validation.json` flags any features that provide zero variance, preventing useless computational overhead.
