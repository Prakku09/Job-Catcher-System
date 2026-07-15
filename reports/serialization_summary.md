# Model Serialization Summary

- **Model Saved As**: `models/job_match_pipeline_v1.0.pkl`
- **File Size**: 195.58 KB
- **Verification Status**: PASS

## Metadata
```json
{
  "model_name": "Job-Catcher-Match-Pipeline",
  "model_version": "1.0",
  "training_timestamp": "2026-07-15T09:38:04.337709Z",
  "git_commit": "84cf91f",
  "dataset_name": "clean_modelling_table.csv",
  "n_training_samples": 425,
  "selected_features": [
    "years_experience",
    "exp_required_years",
    "python_score",
    "python_required",
    "sql_score",
    "sql_required",
    "ml_score",
    "ml_required",
    "javascript_score",
    "javascript_required",
    "data_structures_score",
    "data_structures_required",
    "statistics_score",
    "statistics_required",
    "expected_salary_inr",
    "salary_offered_inr",
    "education_level",
    "edu_minimum",
    "location_student",
    "location_job"
  ],
  "model_type": "Pipeline(JobMatchPreprocessor -> StandardScaler -> XGBClassifier)",
  "hyperparameters": {
    "objective": "binary:logistic",
    "base_score": null,
    "booster": "dart",
    "callbacks": null,
    "colsample_bylevel": null,
    "colsample_bynode": null,
    "colsample_bytree": 0.9581244358047011,
    "device": null,
    "early_stopping_rounds": null,
    "enable_categorical": true,
    "eval_metric": "logloss",
    "feature_types": null,
    "feature_weights": null,
    "gamma": null,
    "grow_policy": null,
    "importance_type": null,
    "interaction_constraints": null,
    "learning_rate": 0.06643907262316669,
    "max_bin": null,
    "max_cat_threshold": null,
    "max_cat_to_onehot": null,
    "max_delta_step": null,
    "max_depth": 4,
    "max_leaves": null,
    "min_child_weight": 4,
    "missing": NaN,
    "monotone_constraints": null,
    "multi_strategy": null,
    "n_estimators": 189,
    "n_jobs": null,
    "num_parallel_tree": null,
    "random_state": 42,
    "reg_alpha": null,
    "reg_lambda": null,
    "sampling_method": null,
    "scale_pos_weight": null,
    "subsample": 0.5018208586863979,
    "tree_method": null,
    "validate_parameters": null,
    "verbosity": null,
    "lambda": 4.7466317986259917e-07,
    "alpha": 0.002458768347634345
  },
  "training_metrics": {
    "accuracy": 0.76,
    "precision": 0.72,
    "recall": 0.6206896551724138,
    "f1": 0.6666666666666666,
    "roc_auc": 0.8553223388305847,
    "pr_auc": 0.753184200467869
  },
  "environment": {
    "python_version": "3.13.14",
    "sklearn_version": "1.9.0",
    "joblib_version": "1.5.3"
  },
  "random_seed": 42
}
```
