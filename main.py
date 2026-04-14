import pandas as pd
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.analyzer import load_image_data, perform_comprehensive_analysis
from src.visualizer import generate_all_image_charts

def generate_report(analysis_results, output_file='output/image_analysis_report.txt'):
    os.makedirs('output', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Image Feature Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("BASIC STATISTICS\n")
        f.write("-" * 40 + "\n")
        basic_stats = analysis_results['basic_statistics']
        for feature, stats in basic_stats.items():
            f.write(f"{feature}:\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Median: {stats['median']:.2f}\n")
            f.write(f"  Std: {stats['std']:.2f}\n")
            f.write(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n\n")
        
        f.write("CONTRAST DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        contrast = analysis_results['contrast_distribution']
        f.write(f"Low Contrast: {contrast['low']}\n")
        f.write(f"Medium Contrast: {contrast['medium']}\n")
        f.write(f"High Contrast: {contrast['high']}\n")
        f.write(f"Mean: {contrast['mean']:.4f}\n\n")
        
        f.write("BRIGHTNESS DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        brightness = analysis_results['brightness_distribution']
        f.write(f"Dark Images: {brightness['dark']}\n")
        f.write(f"Medium Images: {brightness['medium']}\n")
        f.write(f"Bright Images: {brightness['bright']}\n")
        f.write(f"Mean Brightness: {brightness['mean_brightness']:.2f}\n\n")
        
        f.write("TEXTURE PROPERTIES\n")
        f.write("-" * 40 + "\n")
        texture = analysis_results['texture_properties']
        for metric, stats in texture.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n\n")
        
        f.write("SHAPE FEATURES\n")
        f.write("-" * 40 + "\n")
        shape = analysis_results['shape_features']
        f.write(f"Area - Mean: {shape['area_stats']['mean']:.2f}, Std: {shape['area_stats']['std']:.2f}\n")
        f.write(f"Corner Count - Mean: {shape['corner_stats']['mean']:.2f}, Std: {shape['corner_stats']['std']:.2f}\n\n")
        
        f.write("OUTLIER DETECTION\n")
        f.write("-" * 40 + "\n")
        contrast_outliers = analysis_results['contrast_outliers']
        f.write(f"Contrast Outliers: {contrast_outliers['outlier_count']}\n")
        entropy_outliers = analysis_results['entropy_outliers']
        f.write(f"Entropy Outliers: {entropy_outliers['outlier_count']}\n\n")
        
        f.write("CLUSTERING RESULTS\n")
        f.write("-" * 40 + "\n")
        cluster = analysis_results['clustering']
        unique, counts = np.unique(cluster['cluster_labels'], return_counts=True)
        for label, count in zip(unique, counts):
            f.write(f"Cluster {label}: {count} images\n")
        f.write(f"Inertia: {cluster['inertia']:.2f}\n")
    
    print(f"Report saved to {output_file}")

def main():
    print("Image Feature Analysis System v1.0")
    print("=" * 40)
    
    os.makedirs('output', exist_ok=True)
    
    df = load_image_data('data/image_features.csv')
    print(f"Loaded {len(df)} image records")
    print(f"Features: {list(df.columns)}")
    
    print("\nPerforming analysis...")
    results = perform_comprehensive_analysis(df)
    
    print("\nBasic Statistics:")
    basic_stats = results['basic_statistics']
    for feature, stats in basic_stats.items():
        print(f"  {feature}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
    
    print("\nContrast Distribution:")
    contrast = results['contrast_distribution']
    print(f"  Low: {contrast['low']}, Medium: {contrast['medium']}, High: {contrast['high']}")
    
    print("\nBrightness Distribution:")
    brightness = results['brightness_distribution']
    print(f"  Dark: {brightness['dark']}, Medium: {brightness['medium']}, Bright: {brightness['bright']}")
    
    print("\nClustering Results:")
    cluster = results['clustering']
    unique, counts = np.unique(cluster['cluster_labels'], return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Cluster {label}: {count} images")
    
    print("\nGenerating charts...")
    generate_all_image_charts(results)
    
    print("\nGenerating report...")
    generate_report(results)
    
    print("\nAnalysis complete!")
    print("Output files saved to 'output/' directory")

if __name__ == '__main__':
    main()
