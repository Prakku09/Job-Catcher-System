import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

def create_directories():
    for d in ['artifacts', 'plots', 'models']:
        os.makedirs(d, exist_ok=True)

def generate_business_name(profile_row):
    """Dynamically assigns a business name based on mean feature characteristics."""
    # Basic logic for demonstration. Can be fine-tuned based on exact feature distributions.
    if profile_row.get('years_experience', 0) > 4:
        return "Senior Technical Experts"
    elif profile_row.get('expected_salary_inr', 0) > 1500000:
        return "High-Value Candidates"
    elif profile_row.get('python_score', 0) > 7 and profile_row.get('ml_score', 0) > 6:
        return "Specialized Data/ML Talent"
    elif profile_row.get('years_experience', 0) <= 2:
        return "Junior Entry-Level Talent"
    else:
        return "Mid-Level Generalists"

def generate_business_recommendation(business_name):
    if "Senior" in business_name:
        return "Fast-track for leadership roles; prioritize direct interviews with senior engineering managers."
    elif "High-Value" in business_name:
        return "Ensure competitive compensation packages and highlight company culture to win them."
    elif "Data/ML" in business_name:
        return "Target for specialized AI roles; emphasize challenging technical problems in the offer."
    elif "Junior" in business_name:
        return "Funnel into graduate programs or bootcamps; focus on high-volume automated assessments."
    else:
        return "Standard interview pipeline; evaluate for general software engineering positions."

def main():
    create_directories()
    
    # 1. Load Artifacts from Task 14
    data_path = 'artifacts/prepared_cluster_dataset.csv'
    params_path = 'artifacts/cluster_parameters.json'
    features_path = 'artifacts/selected_features.json'
    
    if not (os.path.exists(data_path) and os.path.exists(params_path)):
        raise FileNotFoundError("Task 14 artifacts missing. Please ensure prepare_clustering.py ran successfully.")
        
    df_scaled = pd.read_csv(data_path)
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    optimal_k = params['optimal_k']
    
    with open(features_path, 'r') as f:
        features_dict = json.load(f)
    features = features_dict['features']
    
    # Load unscaled raw data for business profiling (imputing identically as Task 14)
    df_raw = pd.read_csv('src/data/clean_modelling_table.csv')
    df_unscaled = pd.DataFrame(index=df_raw.index)
    for col in features:
        if col in df_raw.columns:
            df_unscaled[col] = df_raw[col].fillna(df_raw[col].median())
        else:
            df_unscaled[col] = 0.0
    df_unscaled = df_unscaled.drop_duplicates().reset_index(drop=True)
    
    # 2. Train KMeans Model
    print(f"Training K-Means with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    # Save the model
    joblib.dump(kmeans, 'artifacts/kmeans_model.pkl')
    
    # Save assignments
    assignments_df = df_unscaled.copy()
    assignments_df['cluster'] = cluster_labels
    assignments_df.to_csv('artifacts/cluster_assignments.csv', index=False)
    
    # 3. Evaluate Clustering
    sil_score = silhouette_score(df_scaled, cluster_labels)
    wcss = kmeans.inertia_
    ch_score = calinski_harabasz_score(df_scaled, cluster_labels)
    db_score = davies_bouldin_score(df_scaled, cluster_labels)
    
    metrics = {
        "silhouette_score": float(sil_score),
        "inertia_wcss": float(wcss),
        "calinski_harabasz_index": float(ch_score),
        "davies_bouldin_index": float(db_score),
        "optimal_k": optimal_k,
        "n_samples": len(df_scaled)
    }
    
    with open('artifacts/cluster_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 4. Cluster Profiling & Business Naming
    profile_list = []
    
    # Calculate means
    cluster_means = assignments_df.groupby('cluster').mean()
    
    for cluster_id in range(optimal_k):
        subset = assignments_df[assignments_df['cluster'] == cluster_id]
        row_mean = cluster_means.loc[cluster_id]
        
        biz_name = generate_business_name(row_mean)
        biz_rec = generate_business_recommendation(biz_name)
        
        profile = {
            "Cluster_ID": int(cluster_id),
            "Business_Name": biz_name,
            "Number_of_Samples": len(subset),
            "Recommendation": biz_rec
        }
        
        # Add summary stats for all features
        for col in features:
            profile[f"Mean_{col}"] = round(float(row_mean[col]), 2)
            
        profile_list.append(profile)
        
    profile_df = pd.DataFrame(profile_list)
    profile_df.to_csv('artifacts/cluster_profiles.csv', index=False)
    
    print("\nCluster Profiles Generated:")
    print(profile_df[['Cluster_ID', 'Business_Name', 'Number_of_Samples']])
    
    # 5. Stability Validation (Multiple Random Seeds)
    print("\nValidating Stability Across Random Seeds...")
    seeds = [42, 52, 62, 72, 82]
    base_labels = cluster_labels
    
    stability_results = {}
    
    for seed in seeds:
        if seed == 42:
            continue # Baseline
        km = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, random_state=seed)
        test_labels = km.fit_predict(df_scaled)
        
        ari = adjusted_rand_score(base_labels, test_labels)
        nmi = normalized_mutual_info_score(base_labels, test_labels)
        sil = silhouette_score(df_scaled, test_labels)
        
        stability_results[f"seed_{seed}"] = {
            "ARI": float(ari),
            "NMI": float(nmi),
            "silhouette_variation": float(abs(sil - sil_score))
        }
    
    with open('artifacts/stability_metrics.json', 'w') as f:
        json.dump(stability_results, f, indent=4)
        
    # Validation checks
    validation = {
        "no_empty_clusters": all(profile_df['Number_of_Samples'] > 0),
        "cluster_sizes_reasonable": all(profile_df['Number_of_Samples'] >= len(df_scaled) * 0.05), # At least 5% of data
        "all_samples_assigned": sum(profile_df['Number_of_Samples']) == len(df_scaled),
        "metrics_computed": True
    }
    
    with open('artifacts/cluster_validation.json', 'w') as f:
        json.dump(validation, f, indent=4)
        
    # 6. Visualizations
    
    # PCA Scatter
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(df_scaled)
    
    plt.figure(figsize=(10, 6))
    for cluster_id in range(optimal_k):
        biz_name = profile_df.loc[profile_df['Cluster_ID'] == cluster_id, 'Business_Name'].values[0]
        mask = cluster_labels == cluster_id
        plt.scatter(pca_2d[mask, 0], pca_2d[mask, 1], label=f'{biz_name} (ID {cluster_id})', alpha=0.6)
    
    plt.title('Cluster Scatter Plot (2D PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/cluster_scatter.png')
    plt.close()
    
    # Cluster Centroid Visualization
    centroids = kmeans.cluster_centers_
    plt.figure(figsize=(12, 6))
    sns.heatmap(centroids, cmap='coolwarm', xticklabels=features, yticklabels=[f"Cluster {i}" for i in range(optimal_k)])
    plt.title('Cluster Centroid Visualization (Scaled Features)')
    plt.tight_layout()
    plt.savefig('plots/cluster_centroids.png')
    plt.close()
    
    # Cluster Size Distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Cluster_ID', y='Number_of_Samples', data=profile_df, palette='viridis')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig('plots/cluster_sizes.png')
    plt.close()
    
    # Feature Distribution by Cluster (Boxplot for Salary & Experience)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='cluster', y='years_experience', data=assignments_df, palette='Set2')
    plt.title('Experience Distribution by Cluster')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='cluster', y='expected_salary_inr', data=assignments_df, palette='Set2')
    plt.title('Expected Salary Distribution by Cluster')
    plt.tight_layout()
    plt.savefig('plots/feature_distributions.png')
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_scaled.corr(), annot=False, cmap='magma')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    print("\nTask 15 completed successfully! All artifacts and plots saved.")

if __name__ == '__main__':
    main()
