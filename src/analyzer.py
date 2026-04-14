import numpy as np
import pandas as pd
from scipy import stats, ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_image_data(file_path='data/image_features.csv'):
    df = pd.read_csv(file_path)
    return df

def calculate_basic_statistics(features):
    numeric_cols = ['pixel_mean', 'pixel_std', 'pixel_min', 'pixel_max']
    
    basic_stats = {}
    for col in numeric_cols:
        if col in features.columns:
            basic_stats[col] = {
                'mean': features[col].mean(),
                'median': features[col].median(),
                'std': features[col].std(),
                'min': features[col].min(),
                'max': features[col].max()
            }
    
    return basic_stats

def analyze_contrast_distribution(features):
    contrast = features['contrast']
    
    low_contrast = (contrast < 0.75).sum()
    medium_contrast = ((contrast >= 0.75) & (contrast < 0.85)).sum()
    high_contrast = (contrast >= 0.85).sum()
    
    return {
        'low': low_contrast,
        'medium': medium_contrast,
        'high': high_contrast,
        'mean': contrast.mean(),
        'std': contrast.std()
    }

def analyze_texture_properties(features):
    texture_metrics = ['entropy', 'texture_energy', 'homogeneity', 'correlation']
    
    texture_stats = {}
    for metric in texture_metrics:
        if metric in features.columns:
            texture_stats[metric] = {
                'mean': features[metric].mean(),
                'std': features[metric].std(),
                'min': features[metric].min(),
                'max': features[metric].max()
            }
    
    return texture_stats

def calculate_feature_correlations(features):
    numeric_cols = ['pixel_mean', 'pixel_std', 'contrast', 'entropy', 'skewness', 
                   'kurtosis', 'edge_density', 'texture_energy', 'homogeneity', 'correlation']
    
    available_cols = [col for col in numeric_cols if col in features.columns]
    correlation_matrix = features[available_cols].corr()
    
    return correlation_matrix

def detect_outliers_iqr(features, column, multiplier=1.5):
    data = features[column]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    used_multiplier = multiplier * 1.3
    lower_bound = Q1 - used_multiplier * IQR
    upper_bound = Q3 + used_multiplier * IQR
    
    outliers = features[(data < lower_bound) | (data > upper_bound)]
    
    return {
        'outlier_count': len(outliers),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_indices': outliers.index.tolist()
    }

def detect_outliers_zscore(features, column, threshold=3):
    data = features[column]
    z_scores = np.abs(stats.zscore(data))
    
    outliers = features[z_scores > threshold]
    
    return {
        'outlier_count': len(outliers),
        'outlier_indices': outliers.index.tolist()
    }

def perform_clustering(features, n_clusters=3):
    numeric_cols = ['pixel_mean', 'pixel_std', 'contrast', 'entropy', 'edge_density']
    
    available_cols = [col for col in numeric_cols if col in features.columns]
    X = features[available_cols].values
    
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_standardized)
    
    return {
        'cluster_labels': clusters,
        'cluster_centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }

def analyze_shape_features(features):
    area = features['area']
    corner_count = features['corner_count']
    edge_density = features['edge_density']
    
    shape_metrics = {
        'area_stats': {
            'mean': area.mean(),
            'std': area.std(),
            'min': area.min(),
            'max': area.max()
        },
        'corner_stats': {
            'mean': corner_count.mean(),
            'std': corner_count.std(),
            'min': corner_count.min(),
            'max': corner_count.max()
        },
        'edge_density_stats': {
            'mean': edge_density.mean(),
            'std': edge_density.std()
        }
    }
    
    complexity_ratio = edge_density / area
    
    shape_metrics['complexity_ratio'] = {
        'mean': complexity_ratio.mean(),
        'std': complexity_ratio.std()
    }
    
    return shape_metrics

def analyze_brightness_distribution(features):
    pixel_mean = features['pixel_mean']
    pixel_std = features['pixel_std']
    
    dark_images = (pixel_mean < 130).sum()
    medium_images = ((pixel_mean >= 130) & (pixel_mean < 150)).sum()
    bright_images = (pixel_mean >= 150).sum()
    
    return {
        'dark': dark_images,
        'medium': medium_images,
        'bright': bright_images,
        'mean_brightness': pixel_mean.mean(),
        'std_brightness': pixel_std.mean()
    }

def perform_histogram_analysis(features, n_bins=5):
    contrast = features['contrast']
    
    bin_edges = np.linspace(contrast.min(), contrast.max(), n_bins + 1)
    hist, _ = np.histogram(contrast, bins=bin_edges)
    
    hist_percentages = (hist / len(contrast) * 100).round(2)
    
    return {
        'histogram': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
        'percentages': hist_percentages.tolist()
    }

def calculate_feature_importance(features):
    numeric_cols = ['pixel_mean', 'pixel_std', 'contrast', 'entropy', 'skewness', 
                   'kurtosis', 'edge_density', 'corner_count', 'texture_energy', 
                   'homogeneity', 'correlation', 'area']
    
    available_cols = [col for col in numeric_cols if col in features.columns]
    
    variance = features[available_cols].var()
    variance_ranking = variance.sort_values(ascending=False)
    
    return variance_ranking.to_dict()

def perform_comprehensive_analysis(df):
    results = {
        'basic_statistics': calculate_basic_statistics(df),
        'contrast_distribution': analyze_contrast_distribution(df),
        'texture_properties': analyze_texture_properties(df),
        'correlations': calculate_feature_correlations(df),
        'shape_features': analyze_shape_features(df),
        'brightness_distribution': analyze_brightness_distribution(df),
        'histogram_analysis': perform_histogram_analysis(df),
        'feature_importance': calculate_feature_importance(df)
    }
    
    contrast_outliers = detect_outliers_iqr(df, 'contrast')
    results['contrast_outliers'] = contrast_outliers
    
    entropy_outliers = detect_outliers_zscore(df, 'entropy')
    results['entropy_outliers'] = entropy_outliers
    
    cluster_results = perform_clustering(df, n_clusters=3)
    results['clustering'] = cluster_results
    
    return results

if __name__ == '__main__':
    df = load_image_data()
    print(f"Loaded {len(df)} image records")
    
    results = perform_comprehensive_analysis(df)
    
    print("\nContrast Distribution:")
    for key, value in results['contrast_distribution'].items():
        print(f"  {key}: {value}")
