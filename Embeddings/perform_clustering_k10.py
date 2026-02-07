"""
Perform fine-grained semantic clustering (k=10) for reader exploration.

This script:
1. Loads semantic embeddings (653 × 384)
2. Performs K-means clustering (k=10)
3. Identifies representative headlines per cluster
4. Generates reader-oriented theme descriptions
5. Saves results independently from k=3 clustering
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
    print("LOADING DATA FOR FINE-GRAINED CLUSTERING (k=10)")
    print("=" * 70)
    
    # Load embeddings
    print("\nLoading embeddings...")
    embeddings = np.load('News_Embedding_True.npy')
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Load news data
    print("\nLoading news articles...")
    df = pd.read_csv('news.csv')
    print(f"✓ Articles loaded: {len(df)}")
    
    return embeddings, df


def perform_clustering(embeddings, n_clusters=10):
    """Perform K-means clustering for fine-grained analysis."""
    print("\n" + "=" * 70)
    print("PERFORMING K-MEANS CLUSTERING (k=10)")
    print("=" * 70)
    
    print(f"\nParameters:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Goal: Discover fine-grained themes for reader exploration")
    print(f"  Algorithm: K-means++")
    
    # K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=30,  # More random starts for better stability with k=10
        max_iter=500,
        random_state=42,
        verbose=0
    )
    
    print(f"\nFitting K-means with {n_clusters} clusters...")
    cluster_labels = kmeans.fit_predict(embeddings)
    
    print(f"✓ Clustering complete")
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("CLUSTER QUALITY METRICS")
    print("=" * 70)
    
    inertia = kmeans.inertia_
    silhouette = silhouette_score(embeddings, cluster_labels)
    
    print(f"\n✓ Inertia: {inertia:.2f}")
    print(f"✓ Silhouette score: {silhouette:.4f}")
    
    # Cluster sizes
    print(f"\n✓ Cluster size distribution:")
    for i in range(n_clusters):
        count = np.sum(cluster_labels == i)
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {i}: {count:3d} articles ({percentage:5.1f}%)")
    
    return kmeans, cluster_labels, silhouette


def get_representative_headlines(embeddings, df, kmeans, cluster_labels, n_representatives=8):
    """Get representative headlines for each cluster."""
    print("\n" + "=" * 70)
    print("IDENTIFYING REPRESENTATIVE HEADLINES")
    print("=" * 70)
    
    representatives = {}
    
    for cluster_id in range(kmeans.n_clusters):
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
        
        print(f"  Cluster {cluster_id}: {top_n} representatives selected")
    
    return representatives


def generate_reader_oriented_themes(representatives):
    """Generate reader-oriented theme descriptions based on headlines."""
    print("\n" + "=" * 70)
    print("ANALYZING CLUSTER THEMES FOR READER INTEREST")
    print("=" * 70)
    
    theme_descriptions = {}
    
    for cluster_id, cluster_df in representatives.items():
        print(f"\n[Cluster {cluster_id}] - {len(cluster_df)} articles")
        
        # Show sample headlines for manual theme identification
        print("Sample headlines:")
        for idx, row in cluster_df.head(5).iterrows():
            headline = row['Executive Headline'][:75]
            print(f"  • {headline}...")
    
    return theme_descriptions


def save_results(df, cluster_labels, kmeans, representatives, silhouette):
    """Save clustering results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Add cluster assignments to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Fine_Grained_Cluster_ID'] = cluster_labels
    
    # Calculate distance to centroid for each article
    embeddings = np.load('News_Embedding_True.npy')
    distances = []
    for i, cluster_id in enumerate(cluster_labels):
        centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        dist = cdist(embeddings[i].reshape(1, -1), centroid, metric='euclidean')[0, 0]
        distances.append(dist)
    
    df_with_clusters['Distance_to_Centroid'] = distances
    
    # Save to CSV
    output_csv = 'news_with_clusters_k10.csv'
    df_with_clusters.to_csv(output_csv, index=False)
    print(f"\n✓ Saved: {output_csv}")
    
    # Create analysis report
    report_lines = []
    report_lines.append("# Fine-Grained Semantic Clustering Analysis (k=10)")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(f"- **Total articles:** {len(df)}")
    report_lines.append(f"- **Number of clusters:** {kmeans.n_clusters}")
    report_lines.append(f"- **Clustering method:** K-means (k-means++)")
    report_lines.append(f"- **Goal:** Discover fine-grained themes for reader exploration")
    report_lines.append("")
    
    report_lines.append("## Cluster Quality Metrics")
    report_lines.append("")
    report_lines.append(f"- **Silhouette score:** {silhouette:.4f}")
    report_lines.append(f"- **Inertia:** {kmeans.inertia_:.2f}")
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
    
    # Add clusters analysis
    report_lines.append("## Cluster Themes (Reader-Oriented)")
    report_lines.append("")
    report_lines.append("_Each cluster represents a distinct thematic group for reader exploration._")
    report_lines.append("")
    
    for cluster_id, cluster_df in representatives.items():
        count = np.sum(cluster_labels == cluster_id)
        report_lines.append(f"### Cluster {cluster_id}")
        report_lines.append("")
        report_lines.append(f"**Size:** {count} articles")
        report_lines.append("")
        report_lines.append("**Representative Headlines:**")
        report_lines.append("")
        
        for idx, row in cluster_df.head(8).iterrows():
            headline = row['Executive Headline']
            report_lines.append(f"- {headline}")
        
        report_lines.append("")
        report_lines.append("**Theme Description:**")
        report_lines.append("")
        report_lines.append("_[To be filled based on headline analysis - focus on reader interest]_")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Save report
    report_file = 'cluster_analysis_k10.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Saved: {report_file}")
    
    return output_csv, report_file


def main():
    """Main execution."""
    print("=" * 70)
    print("FINE-GRAINED SEMANTIC CLUSTERING (k=10)")
    print("Discovering detailed themes for reader exploration")
    print("=" * 70)
    
    # Load data
    embeddings, df = load_data()
    
    # Perform clustering
    kmeans, cluster_labels, silhouette = perform_clustering(embeddings, n_clusters=10)
    
    # Get representatives
    representatives = get_representative_headlines(embeddings, df, kmeans, cluster_labels)
    
    # Analyze themes
    themes = generate_reader_oriented_themes(representatives)
    
    # Save results
    output_csv, report_file = save_results(df, cluster_labels, kmeans, representatives, silhouette)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓✓✓ FINE-GRAINED CLUSTERING COMPLETE ✓✓✓")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  1. {output_csv}")
    print(f"     - Original data + Fine_Grained_Cluster_ID (0-9)")
    print(f"     - Distance_to_Centroid for each article")
    print(f"\n  2. {report_file}")
    print(f"     - Quality metrics")
    print(f"     - Representative headlines per cluster")
    print(f"     - Theme descriptions (to be manually completed)")
    print(f"\nNext steps:")
    print(f"  1. Review {report_file}")
    print(f"  2. Add reader-oriented theme descriptions")
    print(f"  3. Use for content discovery and exploration")
    print()


if __name__ == '__main__':
    main()
