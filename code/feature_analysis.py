import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os


class feature_analysis:
    """A class to explore patterns in frequency band powers that were computed by the EEGPreprocessor class"""

    def __init__(self, data_path: str = '/Users/arielmotsenyat/Documents/coding-workspace/ChronicPainClassifier/data/processed_data_rel_all_pain.csv'):
        """Initialize data path"""
        self.data = self._load_data(data_path)
        self.results_path = self._create_results_directory()
        self.frequency_bands = {
            'Delta': (1, 3),
            'Theta': (3, 7),
            'Alpha': (8, 13),
            'Beta': (13, 30)
        }
        self.regions = ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal']

    def _load_data(self, data_path: str) ->pd.DataFrame:
        """Load EEG data from CSV file"""
        try:
            data = pd.read_csv(data_path)
            required_columns = ['Condition']
            if not all(col in data.columns for col in required_columns):
                raise ValueError("Missing required columns in data")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
    def _create_results_directory(self) -> Path:
        """Create timestamped directory for results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('results') / timestamp
        os.makedirs(results_dir / 'plots' , exist_ok=True)
        os.makedirs(results_dir / 'stats' , exist_ok=True)
        return results_dir
    
    def _get_band_columns(self, band: str) -> list:
        """Get columns related to specific frequency band"""
        band_cols = [col for col in self.data.columns if band in col]
        return band_cols
    
    def _save_plot(self, plt, name: str):
        """Save plot to results directory"""
        plt.savefig(self.results_path / 'plots' / f'{name}.png')
        plt.close()
    
    def plot_band_distributions(self, band: str):
        """Create distribution plot"""
        band_cols = self._get_band_columns(band)

        plt.figure(figsize=(15, 10))
        for col in band_cols[:5]:
            sns.kdeplot(data=self.data, x=col, hue='Condition')
        plt.title(f'{band} Band Power Distribution by Condition')
        plt.xlabel('Power')
        plt.ylabel('Density')
        self._save_plot(plt, f'{band}_distribution')
    
    def plot_rel_comparisons(self):
        """Create plots comparing relative power across frequency bands between groups."""
        plt.figure(figsize=(15, 10))
        
        # Collect data for all bands
        plot_data = []
        for band in self.frequency_bands.keys():
            rel_cols = [col for col in self.data.columns 
                        if f'{band}_total_rel' in col]
            
            if rel_cols:
                # Melt data for this band
                band_data = self.data.melt(
                    id_vars=['Condition'], 
                    value_vars=rel_cols,
                    var_name='Channel',
                    value_name='Power'
                )
                band_data['Band'] = band
                plot_data.append(band_data)
        
        if not plot_data:
            print("No relative power data found")
            return
        
        # Combine all bands into a single DataFrame
        combined_data = pd.concat(plot_data, ignore_index=True)
        
        # Create boxplot with bands on x-axis and condition as hue
        sns.boxplot(data=combined_data, x='Band', y='Power', hue='Condition')
        plt.title('Relative Power Comparison Across Frequency Bands')
        plt.ylabel('Power')
        plt.legend(title='Condition')
        self._save_plot(plt, 'relative_power_comparison')

    def plot_regional_comparisons(self):
        """Create plots comparing relative power between groups."""
        for region in self.regions:
            plt.figure(figsize=(15, 10))
            
            # Collect data for all bands in this region
            plot_data = []
            for band in self.frequency_bands.keys():
                regional_cols = [col for col in self.data.columns 
                            if region in col and band in col]
                
                if regional_cols:
                    # Melt data for this band
                    band_data = self.data.melt(
                        id_vars=['Condition'], 
                        value_vars=regional_cols,
                        var_name='Channel',
                        value_name='Power'
                    )
                    band_data['Band'] = band
                    plot_data.append(band_data)
            
            if not plot_data:
                continue
            
            # Combine all bands for this freq band
            combined_data = pd.concat(plot_data, ignore_index=True)

            # Create boxplot with bands on x-axis and condition as hue
            sns.boxplot(data=combined_data, x='Band', y='Power', hue='Condition')
            plt.title(f'{region} Region Power Comparison Across Frequency Bands')
            plt.ylabel('Power')
            plt.legend(title='Condition')
            self._save_plot(plt, f'{region}_comparison')

    def plot_TAR_analysis(self):
        """Create visualizations for Theta/Alpha ratio analysis."""
        tar_cols = [col for col in self.data.columns if 'theta_alpha_ratio' in col]
    
        if not tar_cols:
            return
        
        for i, col in enumerate(tar_cols[:5]):
            plt.subplots(figsize=(10, 6))
            sns.boxplot(data=self.data, x='Condition', y=col)
            plt.title(f'Theta/Alpha Ratio by Condition - {col}')
            plt.ylabel('TAR')
            self._save_plot(plt, f'TAR_{i}_{col.replace("/", "_")}')

    def plot_coherence_patterns(self):
        """Analyze and visualize coherence patterns."""
        coherence_cols = [col for col in self.data.columns if any(
            f'_{region}_rel' in col for region in self.regions)]
        
        if coherence_cols:
            for condition in self.data['Condition'].unique():
                condition_data = self.data[self.data['Condition'] == condition]
                
                plt.figure(figsize=(20, 15))
                sns.heatmap(
                    condition_data[coherence_cols].corr(), 
                    cmap='coolwarm',
                    center=0
                )
                plt.title(f'Coherence Pattern Correlation Matrix - {condition}')
                self._save_plot(plt, f'coherence_patterns_{condition}')

    def generate_summary_stats(self):
        """Generate summary statistics for all analyses."""
        stats = {}
        
        # Basic group statistics
        stats['group_counts'] = self.data['Condition'].value_counts().to_dict()
        
        # Band power statistics
        for band in self.frequency_bands.keys():
            band_cols = self._get_band_columns(band)
            stats[f'{band}_stats'] = self.data[band_cols].describe().to_dict()
        
        # Save statistics
        pd.DataFrame(stats).to_csv(self.results_path / 'stats' / 'summary_stats.csv')

    def run_full_analysis(self):
        """Run complete analysis pipeline and save all results."""
        # Run all analyses
        for band in self.frequency_bands.keys():
            self.plot_band_distributions(band)
        
        self.plot_regional_comparisons()
        self.plot_rel_comparisons()
        self.plot_TAR_analysis()
        self.plot_coherence_patterns()
        self.generate_summary_stats()

if __name__ == "__main__":
    explorer = feature_analysis()
    explorer.run_full_analysis()