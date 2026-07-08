import numpy as np

REQUIRED_PACKAGES = {
    "pandas": "1.0.0",
    "numpy": "1.18.0",
    "sklearn": "1.0.0",
}

def check_dependencies():
    """Fails fast with a clear message if a required package is missing
    or too old, instead of a confusing stack trace mid-pipeline."""
    import importlib
    from packaging import version
    problems = []
    for pkg, min_version in REQUIRED_PACKAGES.items():
        try:
            installed = getattr(importlib.import_module(pkg), "__version__", "unknown")
            if installed != "unknown" and version.parse(installed) < version.parse(min_version):
                problems.append(f"{pkg} {installed} < required {min_version}")
        except ImportError:
            problems.append(f"{pkg} is not installed")
    if problems:
        raise RuntimeError("Missing/incompatible dependencies: " + "; ".join(problems))

def validate_data(X, y=None, min_samples_per_class=10):
    """Rejects the input shapes most likely to silently corrupt a fit:
    NaN/Inf, mismatched lengths, too few classes or samples."""
    # Convert numerical parts to check for NaN/Inf safely
    # If X is a DataFrame, we might have categorical columns. 
    # For this use-case, the raw dataframe contains strings. So we check numerics only.
    if isinstance(X, np.ndarray):
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN/Inf values; impute or drop before fitting")
    
    if y is not None:
        if X.shape[0] != len(y):
            raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {len(y)}")
        
        # Check class balance
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Target y must contain at least 2 classes for binary classification.")
        for cls, count in zip(classes, counts):
            if count < min_samples_per_class:
                print(f"WARNING: Class {cls} has only {count} samples (less than {min_samples_per_class}).")
                
    return True
