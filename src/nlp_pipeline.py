import pandas as pd
import numpy as np
import os
import re
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import spacy
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Reusable NLP Pipeline Components
# ---------------------------------------------------------

class JobCatcherNLP:
    def __init__(self):
        logging.info("Initializing NLP Pipeline...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.error("SpaCy model 'en_core_web_sm' not found. Ensure it is downloaded.")
            raise
            
        self.stop_words = self.nlp.Defaults.stop_words
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.is_fit = False
        
        # Predefined skills to extract
        self.known_skills = {
            'python', 'sql', 'machine learning', 'javascript', 'data structures', 
            'statistics', 'ai', 'data analysis', 'deep learning', 'aws', 'cloud', 'backend'
        }

    def preprocess_text(self, text):
        """
        Lowercasing, removing HTML/URLs/Punctuation, removing stop words,
        tokenization, and lemmatization using spaCy.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # 3. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 4. SpaCy processing (tokenization, lemmatization, stop words, punctuation)
        doc = self.nlp(text)
        cleaned_tokens = [
            token.lemma_ for token in doc 
            if not token.is_punct and not token.is_space and not token.is_stop and token.is_alpha
        ]
        
        return " ".join(cleaned_tokens)

    def extract_skills(self, text):
        """
        Extract structured information from text based on keyword matching.
        """
        if not isinstance(text, str):
            return []
            
        text_lower = text.lower()
        extracted = [skill for skill in self.known_skills if skill in text_lower]
        return extracted

    def generate_embeddings(self, texts, method="sbert"):
        """
        Generates embeddings using TF-IDF or Sentence Transformers.
        """
        # Ensure safe input
        safe_texts = [str(t) if pd.notnull(t) else "" for t in texts]
        
        if method == "sbert":
            logging.info("Generating Sentence Transformer embeddings...")
            embeddings = self.sbert_model.encode(safe_texts, show_progress_bar=False)
            return embeddings
            
        elif method == "tfidf":
            logging.info("Generating TF-IDF embeddings...")
            if not self.is_fit:
                embeddings = self.tfidf_vectorizer.fit_transform(safe_texts).toarray()
                self.is_fit = True
            else:
                embeddings = self.tfidf_vectorizer.transform(safe_texts).toarray()
            return embeddings
        else:
            raise ValueError(f"Unknown vectorization method: {method}")

    def compute_similarity(self, embed1, embed2):
        """
        Compute cosine similarity between two sets of embeddings.
        """
        return cosine_similarity(embed1, embed2)
        
    def generate_features(self, df_pairs):
        """
        Takes a dataframe with 'resume_text' and 'jd_text', computes similarities,
        extracts skills, and calculates overlap features.
        """
        logging.info("Preprocessing texts...")
        df_pairs['resume_clean'] = df_pairs['resume_text'].apply(self.preprocess_text)
        df_pairs['jd_clean'] = df_pairs['jd_text'].apply(self.preprocess_text)
        
        logging.info("Extracting skills...")
        df_pairs['resume_skills'] = df_pairs['resume_text'].apply(self.extract_skills)
        df_pairs['jd_skills'] = df_pairs['jd_text'].apply(self.extract_skills)
        
        # Calculate overlap
        def calc_overlap(row):
            res_skills = set(row['resume_skills'])
            jd_skills = set(row['jd_skills'])
            overlap = res_skills.intersection(jd_skills)
            missing = jd_skills - res_skills
            return pd.Series({
                'skill_overlap_count': len(overlap),
                'keyword_coverage_percentage': len(overlap) / len(jd_skills) if len(jd_skills) > 0 else 1.0,
                'missing_required_skills': len(missing)
            })
            
        overlap_features = df_pairs.apply(calc_overlap, axis=1)
        df_pairs = pd.concat([df_pairs, overlap_features], axis=1)
        
        logging.info("Computing SBERT similarities...")
        res_emb_sbert = self.generate_embeddings(df_pairs['resume_clean'].tolist(), method="sbert")
        jd_emb_sbert = self.generate_embeddings(df_pairs['jd_clean'].tolist(), method="sbert")
        df_pairs['sbert_similarity'] = [self.compute_similarity(r.reshape(1, -1), j.reshape(1, -1))[0][0] for r, j in zip(res_emb_sbert, jd_emb_sbert)]
        
        logging.info("Computing TF-IDF similarities...")
        res_emb_tfidf = self.generate_embeddings(df_pairs['resume_clean'].tolist(), method="tfidf")
        jd_emb_tfidf = self.generate_embeddings(df_pairs['jd_clean'].tolist(), method="tfidf")
        df_pairs['tfidf_similarity'] = [self.compute_similarity(r.reshape(1, -1), j.reshape(1, -1))[0][0] for r, j in zip(res_emb_tfidf, jd_emb_tfidf)]
        
        return df_pairs, res_emb_sbert, jd_emb_sbert, res_emb_tfidf, jd_emb_tfidf


# ---------------------------------------------------------
# Execution Flow
# ---------------------------------------------------------
def synthesize_text(df):
    """
    Synthesizes unstructured text from structured columns as they don't exist yet in the dataset.
    """
    logging.info("Synthesizing unstructured Resume and JD text from structured columns...")
    
    def make_resume(row):
        skills = []
        if row.get('python_score', 0) > 0: skills.append('Python')
        if row.get('sql_score', 0) > 0: skills.append('SQL')
        if row.get('ml_score', 0) > 0: skills.append('Machine Learning')
        if row.get('javascript_score', 0) > 0: skills.append('JavaScript')
        if row.get('data_structures_score', 0) > 0: skills.append('Data Structures')
        if row.get('statistics_score', 0) > 0: skills.append('Statistics')
        
        text = f"Highly motivated professional with {row.get('years_experience', 0)} years of experience. "
        text += f"Located in {row.get('location_student', 'Unknown')}. "
        text += f"Education level: {row.get('education_level', 'Unknown')}. "
        text += f"Core competencies include: {', '.join(skills)}. "
        return text

    def make_jd(row):
        req_skills = []
        if row.get('python_required', 0) > 0: req_skills.append('Python')
        if row.get('sql_required', 0) > 0: req_skills.append('SQL')
        if row.get('ml_required', 0) > 0: req_skills.append('Machine Learning')
        if row.get('javascript_required', 0) > 0: req_skills.append('JavaScript')
        if row.get('data_structures_required', 0) > 0: req_skills.append('Data Structures')
        if row.get('statistics_required', 0) > 0: req_skills.append('Statistics')
        
        text = f"We are {row.get('company', 'a leading company')} looking for a {row.get('title', 'Professional')}. "
        text += f"This role is based in {row.get('location_job', 'Unknown')}. "
        text += f"We require at least {row.get('exp_required_years', 0)} years of experience and a minimum education of {row.get('edu_minimum', 'Unknown')}. "
        text += f"Must have expertise in: {', '.join(req_skills)}."
        return text

    df['resume_text'] = df.apply(make_resume, axis=1)
    df['jd_text'] = df.apply(make_jd, axis=1)
    return df

def main():
    os.makedirs('models/sentence_embedding_model', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # 1. Load Data
    data_path = 'src/data/clean_modelling_table.csv'
    df = pd.read_csv(data_path)
    
    # Generate texts
    df = synthesize_text(df)
    
    # 2. Run Pipeline
    pipeline = JobCatcherNLP()
    df_features, res_emb_sbert, jd_emb_sbert, res_emb_tfidf, jd_emb_tfidf = pipeline.generate_features(df)
    
    # 3. Evaluation & Error Analysis
    logging.info("Evaluating NLP features...")
    
    # Using roc_auc to see which similarity score correlates better with is_good_match
    sbert_auc = roc_auc_score(df_features['is_good_match'], df_features['sbert_similarity'])
    tfidf_auc = roc_auc_score(df_features['is_good_match'], df_features['tfidf_similarity'])
    
    logging.info(f"SBERT Similarity ROC-AUC: {sbert_auc:.4f}")
    logging.info(f"TF-IDF Similarity ROC-AUC: {tfidf_auc:.4f}")
    
    best_approach = "Sentence Transformers (SBERT)" if sbert_auc > tfidf_auc else "TF-IDF"
    
    # Error Analysis (False Positives / Negatives based on a threshold)
    # Threshold at median similarity
    thresh = df_features['sbert_similarity'].median()
    df_features['pred_match'] = (df_features['sbert_similarity'] > thresh).astype(int)
    
    fp = len(df_features[(df_features['is_good_match'] == 0) & (df_features['pred_match'] == 1)])
    fn = len(df_features[(df_features['is_good_match'] == 1) & (df_features['pred_match'] == 0)])
    
    evaluation_metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sbert_auc": float(sbert_auc),
        "tfidf_auc": float(tfidf_auc),
        "best_representation": best_approach,
        "error_analysis": {
            "threshold_used": float(thresh),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "note": "A full classification model utilizing these NLP features would yield better performance than thresholding similarity alone."
    }
    
    with open('artifacts/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
        
    # 4. Save Artifacts
    logging.info("Saving artifacts...")
    # Arrays
    np.save('artifacts/resume_embeddings.npy', res_emb_sbert)
    np.save('artifacts/jd_embeddings.npy', jd_emb_sbert)
    
    # CSVs
    df_features[['application_id', 'sbert_similarity', 'tfidf_similarity']].to_csv('artifacts/similarity_scores.csv', index=False)
    df_features[['application_id', 'resume_skills', 'jd_skills']].to_csv('artifacts/extracted_skills.csv', index=False)
    
    nlp_features_cols = ['application_id', 'sbert_similarity', 'tfidf_similarity', 'skill_overlap_count', 'keyword_coverage_percentage', 'missing_required_skills']
    df_features[nlp_features_cols].to_csv('artifacts/nlp_features.csv', index=False)
    
    # Models
    joblib.dump(pipeline.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    # Save the pipeline itself without the heavy huggingface model explicitly, or just the whole thing
    # Note: serializing spacy/sbert can be heavy, we just save the class wrapper state
    pipeline.nlp = None # Clear before pickle to save space, will reload on init
    pipeline.sbert_model = None
    joblib.dump(pipeline, 'models/nlp_pipeline.pkl')
    
    # 5. Visualizations
    logging.info("Generating visualizations...")
    
    # Similarity Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_features, x='sbert_similarity', hue='is_good_match', kde=True, bins=30)
    plt.title('SBERT Cosine Similarity Distribution by Match Status')
    plt.savefig('plots/similarity_distribution.png')
    plt.close()
    
    # Similarity Heatmap (sample of 10x10)
    plt.figure(figsize=(8, 6))
    sample_size = min(10, len(res_emb_sbert))
    sample_sim_matrix = cosine_similarity(res_emb_sbert[:sample_size], jd_emb_sbert[:sample_size])
    sns.heatmap(sample_sim_matrix, annot=True, cmap='Blues', fmt=".2f")
    plt.title('Resume vs JD Similarity Heatmap (Sample)')
    plt.xlabel('Job Descriptions')
    plt.ylabel('Resumes')
    plt.savefig('plots/similarity_heatmap.png')
    plt.close()
    
    # TF-IDF Feature Importance (Top Words)
    plt.figure(figsize=(10, 6))
    feature_names = pipeline.tfidf_vectorizer.get_feature_names_out()
    tfidf_sum = np.sum(res_emb_tfidf, axis=0)
    top_indices = np.argsort(tfidf_sum)[-20:]
    plt.barh([feature_names[i] for i in top_indices], tfidf_sum[top_indices])
    plt.title('Top 20 TF-IDF Keywords in Resumes')
    plt.savefig('plots/tfidf_importance.png')
    plt.close()
    
    # Keyword Frequency
    all_skills = [skill for sublist in df_features['resume_skills'] for skill in sublist]
    skill_counts = Counter(all_skills)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(skill_counts.values()), y=list(skill_counts.keys()))
    plt.title('Frequency of Extracted Skills')
    plt.savefig('plots/keyword_frequency.png')
    plt.close()
    
    # Embedding Visualization (PCA)
    pca = PCA(n_components=2)
    res_pca = pca.fit_transform(res_emb_sbert)
    plt.figure(figsize=(10, 6))
    plt.scatter(res_pca[:, 0], res_pca[:, 1], alpha=0.5, c=df_features['is_good_match'], cmap='coolwarm')
    plt.title('PCA of Resume SBERT Embeddings (colored by match)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('plots/embedding_visualization.png')
    plt.close()
    
    logging.info("Task 18 completed successfully!")

if __name__ == "__main__":
    main()
