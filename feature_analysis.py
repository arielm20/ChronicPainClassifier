import pandas as pd
import numpy as np
import logging
import mne
from mne.preprocessing import ICA
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class EEGFeature_analysis:
     """Class to explore eeg features of the preprocessed eeg data"""
     def __init__(self, data_path: str = '/Users/arielmotsenyat/Documents/coding-workspace/ChronicPainClassifier/clean_data/'):
        """initialize preprocessed data path"""
        self.data = self.load_data(data_path)
        self.results_path = self._create_results_dir()

        self.frequency_bands = {
            'Delta': (1, 3),
            'Theta': (3, 7),
            'Alpha': (8, 13),
            'Beta': (13, 30)
        }

        self.regions = {
            'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
            'Central': ['C3', 'C4', 'Cz'],
            'Parietal': ['P3', 'P4', 'Pz', 'POz'],
            'Occipital': ['O1', 'O2'],
            'Temporal': ['T3', 'T4', 'T5', 'T6']
        }

        def load_data(self, data_path):
            "load preprocessed eeg data"
            data = mne.io.read_raw_fif(data_path)
            return data

        def _create_results_directory(self):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = Path('results') / timestamp
            os.makedirs(results_dir / 'plots', exist_ok=True)
            os.makedirs(results_dir / 'stats', exist_ok=True)
            return results_dir
        
        def _save_plot(self, plt, name: str):
            """Save plot to results directory."""
            plt.savefig(self.results_path / 'plots' / f'{name}.png')
            plt.close()

        def extract_band_features(self) -> pd.DataFrame:
                """Extract frequency band features by region. Returns DataFrame with regional frequency band powers"""
                self.logger.info("Extracting frequency band features...")

                features_dict = {}

                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    self.logger.info(f"Processing {band_name} band ({low_freq}-{high_freq} Hz)...")

                    # Filter data for this band
                    filtered = self.processed_data.copy().filter(
                        low_freq, high_freq,
                        fir_design='firwin',
                        verbose=False
                    )
                    # Get data array shape
                    data = filtered.get_data()
                    # Calculate power for each channel
                    power = np.mean(data**2, axis=1)
                    # Create dictionary for each channel and power values
                    channel_powers = {ch: power[idx] for idx, ch in enumerate(self.processed_data.ch_names)}

                    # Calculate mean power for each brain region
                    for region_name, channels in self.regions.items():
                        # Find channels that exist in both the region and the data
                        available_channels = [ch for ch in channels if ch in self.processed_data.ch_names]

                        if available_channels:
                            region_powers = [channel_powers[ch] for ch in available_channels]
                            features_dict[f'{band_name}_{region_name}_mean'] = np.mean(region_powers)
                        else:
                            self.logger.warning(f"No channels found for {region_name} region")
                    # Calculate overall mean for this band (across all channels)
                    features_dict[f'{band_name}_mean'] = np.mean(power)
                
                # Convert to DataFrame with single row
                features_df = pd.DataFrame([features_dict])
                
                self.logger.info(f"Extracted {len(features_dict)} band features")
                return features_df

        def create_processed_dataset(self, condition: str = None) -> pd.DataFrame:
            """Create the final processed dataset for modeling.
            Args:
                condition: Optionaly include condition label for 'chronic_pain' vs 'healthy_controls'
            """
            self.logger.info("Creating processed dataset...")
            
            # Load and clean data if not already done
            if self.processed_data is None:
                self.clean_data()
                
            # Extract frequency band features
            band_features = self.extract_band_features()

            # Add condition if provided
            if condition is not None:
                band_features['Condition'] = condition
        
            self.processed_data_df = band_features
            self.processed_data_df['Participant #'] = sub_id
            self.logger.info(f"Processed dataset created. Shape: {self.processed_data_df.shape}")
            return self.processed_data_df

        def save_processed_data(self, output_path: str = None):
            """Save processed dataset to file."""
            if output_path is None:
                output_path = '/Users/arielmotsenyat/Documents/coding-workspace/ChronicPainClassifier/data/processed_data.csv'
                
            if self.processed_data_df is None:
                self.create_processed_dataset()
                
            self.processed_data_df.to_csv(output_path, mode='a', header=True, index=False)
            self.logger.info(f"Processed data saved to {output_path}")