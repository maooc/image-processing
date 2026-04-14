import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_distribution(features, save_path='output/feature_distribution.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].hist(features['pixel_mean'], bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Pixel Mean')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Pixel Mean Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(features['pixel_std'], bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Pixel Std')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Pixel Std Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(features['contrast'], bins=20, color='red', alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Contrast')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Contrast Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].hist(features['entropy'], bins=20, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Entropy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Entropy Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(features['edge_density'], bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Edge Density')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Edge Density Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(features['homogeneity'], bins=20, color='cyan', alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Homogeneity')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Homogeneity Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature distribution to {save_path}")

def plot_contrast_distribution(contrast_data, save_path='output/contrast_distribution.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Low (<0.75)', 'Medium (0.75-0.85)', 'High (>0.85)']
    values = [contrast_data['low'], contrast_data['medium'], contrast_data['high']]
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black')
    
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                str(value), ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Contrast Category', fontsize=12)
    ax.set_ylabel('Image Count', fontsize=12)
    ax.set_title('Contrast Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved contrast distribution to {save_path}")

def plot_correlation_heatmap(correlation_matrix, save_path='output/correlation_heatmap.png'):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlation_matrix.columns)
    
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")

def plot_brightness_distribution(brightness_data, save_path='output/brightness_distribution.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Dark (<130)', 'Medium (130-150)', 'Bright (>150)']
    values = [brightness_data['dark'], brightness_data['medium'], brightness_data['bright']]
    
    colors = ['#2c3e50', '#95a5a6', '#f1c40f']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black')
    
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                str(value), ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Brightness Category', fontsize=12)
    ax.set_ylabel('Image Count', fontsize=12)
    ax.set_title('Brightness Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved brightness distribution to {save_path}")

def plot_feature_scatter(features, save_path='output/feature_scatter.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(features['pixel_mean'], features['contrast'], 
                   c='blue', alpha=0.6, edgecolors='black', s=50)
    axes[0].set_xlabel('Pixel Mean', fontsize=12)
    axes[0].set_ylabel('Contrast', fontsize=12)
    axes[0].set_title('Mean vs Contrast', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(features['entropy'], features['edge_density'], 
                   c='green', alpha=0.6, edgecolors='black', s=50)
    axes[1].set_xlabel('Entropy', fontsize=12)
    axes[1].set_ylabel('Edge Density', fontsize=12)
    axes[1].set_title('Entropy vs Edge Density', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature scatter to {save_path}")

def plot_shape_analysis(shape_data, save_path='output/shape_analysis.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    area_stats = shape_data['area_stats']
    categories = ['Min', 'Mean', 'Max']
    values = [area_stats['min'], area_stats['mean'], area_stats['max']]
    
    ax1.bar(categories, values, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Metric', fontsize=12)
    ax1.set_ylabel('Area (pixels)', fontsize=12)
    ax1.set_title('Area Statistics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    corner_stats = shape_data['corner_stats']
    categories2 = ['Min', 'Mean', 'Max']
    values2 = [corner_stats['min'], corner_stats['mean'], corner_stats['max']]
    
    ax2.bar(categories2, values2, color='coral', edgecolor='black')
    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('Corner Count', fontsize=12)
    ax2.set_title('Corner Count Statistics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved shape analysis to {save_path}")

def plot_histogram_analysis(histogram_data, save_path='output/histogram_analysis.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    percentages = histogram_data['percentages']
    bin_edges = histogram_data['bin_edges']
    labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(percentages))]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(percentages)))
    
    bars = ax.bar(labels, percentages, color=colors, edgecolor='black')
    
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{pct}%', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Contrast Range', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Contrast Histogram Analysis', fontsize=14, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram analysis to {save_path}")

def plot_feature_importance(importance_data, save_path='output/feature_importance.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features = list(importance_data.keys())
    variances = list(importance_data.values())
    
    sorted_indices = np.argsort(variances)[::-1]
    features = [features[i] for i in sorted_indices]
    variances = [variances[i] for i in sorted_indices]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))
    
    bars = ax.barh(features[::-1], variances[::-1], color=colors[::-1], edgecolor='black')
    
    ax.set_xlabel('Variance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance (by Variance)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance to {save_path}")

def generate_all_image_charts(analysis_results):
    import os
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(script_dir))
    os.makedirs('output', exist_ok=True)
    
    from src.analyzer import load_image_data
    df = load_image_data()
    
    plot_feature_distribution(df)
    plot_contrast_distribution(analysis_results['contrast_distribution'])
    plot_correlation_heatmap(analysis_results['correlations'])
    plot_brightness_distribution(analysis_results['brightness_distribution'])
    plot_feature_scatter(df)
    plot_shape_analysis(analysis_results['shape_features'])
    plot_histogram_analysis(analysis_results['histogram_analysis'])
    plot_feature_importance(analysis_results['feature_importance'])

if __name__ == '__main__':
    from analyzer import load_image_data, perform_comprehensive_analysis
    
    df = load_image_data()
    results = perform_comprehensive_analysis(df)
    generate_all_image_charts(results)
    print("All image processing charts generated!")
