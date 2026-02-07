"""
Perform semantic clustering on news article embeddings.

This script:
1. Loads semantic embeddings (653 × 384)
2. Performs K-means clustering (k=3)
3. Assigns cluster IDs to each article
4. Identifies representative headlines per cluster
5. Generates neutral theme descriptions
6. Saves results with cluster assignments
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import json
from datetime import datetime


def load_data():
    """Load embeddings and news data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load embeddings
    print("\nLoading embeddings...")
    embeddings = np.load('News_Embedding_True.npy')
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Load news data
    print("\nLoading news articles...")
    df = pd.read_csv('news.csv')
    print(f"✓ Articles loaded: {len(df)}")
    
    # Verify alignment
    if len(embeddings) != len(df):
        print(f"⚠ WARNING: Embedding count ({len(embeddings)}) != Article count ({len(df)})")
    
    return embeddings, df


def perform_clustering(embeddings, n_clusters=3):
    """Perform K-means clustering on embeddings."""
    print("\n" + "=" * 70)
    print("PERFORMING K-MEANS CLUSTERING")
    print("=" * 70)
    
    print(f"\nParameters:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Algorithm: K-means++")
    print(f"  Random state: 42")
    
    # K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=20,
        max_iter=500,
        random_state=42,
        verbose=0
    )
    
    print(f"\nFitting K-means...")
    cluster_labels = kmeans.fit_predict(embeddings)
    
    print(f"✓ Clustering complete")
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("CLUSTER QUALITY METRICS")
    print("=" * 70)
    
    inertia = kmeans.inertia_
    silhouette = silhouette_score(embeddings, cluster_labels)
    
    print(f"\n✓ Inertia (within-cluster sum of squares): {inertia:.2f}")
    print(f"✓ Silhouette score: {silhouette:.4f}")
    print(f"  (Range: -1 to 1, higher is better)")
    
    # Cluster sizes
    print(f"\n✓ Cluster size distribution:")
    for i in range(n_clusters):
        count = np.sum(cluster_labels == i)
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {i}: {count} articles ({percentage:.1f}%)")
    
    return kmeans, cluster_labels, silhouette


def get_representative_headlines(embeddings, df, kmeans, cluster_labels, n_representatives=10):
    """Get representative headlines for each cluster."""
    print("\n" + "=" * 70)
    print("IDENTIFYING REPRESENTATIVE HEADLINES")
    print("=" * 70)
    
    representatives = {}
    
    for cluster_id in range(kmeans.n_clusters):
        print(f"\n[Cluster {cluster_id}]")
        
        # Get articles in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_df = df[cluster_mask].copy()
        
        # Calculate distances to centroid
        centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        distances = cdist(cluster_embeddings, centroid, metric='euclidean').flatten()
        
        # Add distances to dataframe
        cluster_df['distance_to_centroid'] = distances
        
        # Sort by distance (closest first)
        cluster_df = cluster_df.sort_values('distance_to_centroid')
        
        # Get top N representatives
        top_n = min(n_representatives, len(cluster_df))
        representatives[cluster_id] = cluster_df.head(top_n)
        
        print(f"  ✓ Selected {top_n} representative articles")
        print(f"  Distance range: {distances.min():.4f} to {distances.max():.4f}")
    
    return representatives


def analyze_cluster_themes(representatives):
    """Analyze and describe cluster themes."""
    print("\n" + "=" * 70)
    print("ANALYZING CLUSTER THEMES")
    print("=" * 70)
    
    themes = {}
    
    for cluster_id, cluster_df in representatives.items():
        print(f"\n[Cluster {cluster_id}]")
        print(f"Sample headlines:")
        
        # Show top 5 headlines
        for idx, row in cluster_df.head(5).iterrows():
            headline = row['Executive Headline'][:80]
            print(f"  • {headline}...")
        
        # Store for theme generation
        themes[cluster_id] = {
            'headlines': cluster_df['Executive Headline'].tolist(),
            'summaries': cluster_df['Summary'].tolist(),
            'size': len(cluster_df)
        }
    
    return themes


def generate_cluster_descriptions(themes):
    """Generate neutral theme descriptions for each cluster."""
    print("\n" + "=" * 70)
    print("GENERATING THEME DESCRIPTIONS")
    print("=" * 70)
    
    descriptions = {}
    
    # Manual analysis based on representative headlines
    # This will be filled after reviewing the output
    print("\nNote: Theme descriptions will be generated based on manual review")
    print("of representative headlines and summaries.")
    
    for cluster_id in themes.keys():
        descriptions[cluster_id] = f"Cluster {cluster_id} - To be manually described"
    
    return descriptions


def save_results(df, cluster_labels, kmeans, representatives, silhouette):
    """Save clustering results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Add cluster assignments to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Semantic_Cluster_ID'] = cluster_labels
    
    # Calculate distance to centroid for each article
    embeddings = np.load('News_Embedding_True.npy')
    distances = []
    for i, cluster_id in enumerate(cluster_labels):
        centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        dist = cdist(embeddings[i].reshape(1, -1), centroid, metric='euclidean')[0, 0]
        distances.append(dist)
    
    df_with_clusters['Distance_to_Centroid'] = distances
    
    # Save to CSV
    output_csv = 'news_with_clusters.csv'
    df_with_clusters.to_csv(output_csv, index=False)
    print(f"\n✓ Saved: {output_csv}")
    
    # Create analysis report
    report_lines = []
    report_lines.append("# Semantic Clustering Analysis Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(f"- **Total articles:** {len(df)}")
    report_lines.append(f"- **Number of clusters:** {kmeans.n_clusters}")
    report_lines.append(f"- **Clustering method:** K-means (k-means++)")
    report_lines.append(f"- **Embedding dimensions:** 384")
    report_lines.append("")
    
    report_lines.append("## Cluster Quality Metrics")
    report_lines.append("")
    report_lines.append(f"- **Silhouette score:** {silhouette:.4f}")
    report_lines.append(f"  - Range: -1 (poor) to 1 (excellent)")
    report_lines.append(f"  - Interpretation: Measures cluster separation and cohesion")
    report_lines.append(f"- **Inertia:** {kmeans.inertia_:.2f}")
    report_lines.append(f"  - Within-cluster sum of squared distances")
    report_lines.append("")
    
    report_lines.append("## Cluster Distribution")
    report_lines.append("")
    report_lines.append("| Cluster ID | Size | Percentage |")
    report_lines.append("|------------|------|------------|")
    for i in range(kmeans.n_clusters):
        count = np.sum(cluster_labels == i)
        percentage = (count / len(cluster_labels)) * 100
        report_lines.append(f"| {i} | {count} | {percentage:.1f}% |")
    report_lines.append("")
    
    # Add representative headlines for each cluster
    report_lines.append("## Cluster Analysis")
    report_lines.append("")
    
    for cluster_id, cluster_df in representatives.items():
        count = np.sum(cluster_labels == cluster_id)
        report_lines.append(f"### Cluster {cluster_id}")
        report_lines.append("")
        report_lines.append(f"**Size:** {count} articles")
        report_lines.append("")
        report_lines.append("**Representative Headlines:**")
        report_lines.append("")
        
        for idx, row in cluster_df.head(10).iterrows():
            headline = row['Executive Headline']
            report_lines.append(f"- {headline}")
        
        report_lines.append("")
        report_lines.append("**Theme Description:**")
        report_lines.append("")
        report_lines.append("_[To be filled after manual review of headlines]_")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Save report
    report_file = 'cluster_analysis.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Saved: {report_file}")
    
    return output_csv, report_file


def main():
    """Main execution."""
    print("=" * 70)
    print("SEMANTIC CLUSTERING OF NEWS ARTICLES")
    print("=" * 70)
    
    # Load data
    embeddings, df = load_data()
    
    # Perform clustering
    kmeans, cluster_labels, silhouette = perform_clustering(embeddings, n_clusters=3)
    
    # Get representatives
    representatives = get_representative_headlines(embeddings, df, kmeans, cluster_labels)
    
    # Analyze themes
    themes = analyze_cluster_themes(representatives)
    
    # Save results
    output_csv, report_file = save_results(df, cluster_labels, kmeans, representatives, silhouette)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓✓✓ CLUSTERING COMPLETE ✓✓✓")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  1. {output_csv}")
    print(f"     - Original data + Semantic_Cluster_ID (0, 1, 2)")
    print(f"     - Distance_to_Centroid for each article")
    print(f"\n  2. {report_file}")
    print(f"     - Cluster quality metrics")
    print(f"     - Representative headlines per cluster")
    print(f"     - Theme descriptions (for manual completion)")
    print(f"\nNext steps:")
    print(f"  1. Review {report_file}")
    print(f"  2. Add neutral theme descriptions for each cluster")
    print(f"  3. Use cluster assignments for consistent data labeling")
    print()


if __name__ == '__main__':
    main()
