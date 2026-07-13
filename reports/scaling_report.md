# Data Scaling Report

For distance-based clustering algorithms like K-Means, standardizing features is mathematically essential. 
Features like `expected_salary_inr` operate in the hundreds of thousands, while `python_score` operates on a 0-10 scale. Without scaling, Euclidean distance calculations would be completely dominated by the salary feature.

We applied `StandardScaler` to ensure all features have a mean of 0 and a standard deviation of 1, allowing equal geometric contribution across all dimensions.
