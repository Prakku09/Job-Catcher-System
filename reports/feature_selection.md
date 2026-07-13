# Feature Selection for Clustering

The clustering process relies heavily on selecting meaningful dimensions that accurately differentiate candidate profiles. We selected the following numerical features:
- **Technical Skills**: `python_score`, `sql_score`, `ml_score`, `statistics_score`, `data_structures_score`, `javascript_score`. These naturally segment candidates into distinct technical archetypes (e.g., Data Scientists vs Frontend Developers).
- **Experience**: `years_experience`, `exp_required_years`. Experience heavily impacts seniority clustering.
- **Compensation**: `expected_salary_inr`, `salary_offered_inr`. Anchors clusters by compensation bands.

Identifiers (IDs) and the target label (`is_good_match`) are strictly excluded to prevent data leakage and ensure purely unsupervised discovery.
