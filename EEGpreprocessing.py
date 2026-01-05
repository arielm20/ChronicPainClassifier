import pandas as pd
import logging
import mne
from mne.preprocessing import ICA

class EEGPreprocessor:
    """
    Class for pre processing EEG data for identifying frequency band features of chronic pain.
    Extracts frequency information from raw EEG data for AD analysis
    """
    def __init__(self, data_path: str = '/Users/arielmotsenyat/Documents/coding-workspace/mun_pain_data'):
        """initialize preprocessor with correct data path"""
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.logging_setup()

        self.frequency_bands = {
            'Delta': '_1_3',
            'Theta': '3_7',
            'Alpha': '8_13',
            'Beta': '13_30'
        }

        self.regions = {
            'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
            'Central': ['C3', 'C4', 'Cz'],
            'Parietal': ['P3', 'P4', 'Pz', 'POz'],
            'Occipital': ['O1', 'O2'],
            'Temporal': ['T3', 'T4', 'T5', 'T6']
        }
    
    def logging_setup(self):
        """setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self) -> pd.DataFrame:
        """this function will load, filter, apply ICA, and return cleaend EEG data"""
        self.logger.info("Loading data...")
        self.data = mne.io.read_raw_brainvision(self.data_path, preload=True)
        self.data.set_channel_types({ch: "eeg" for ch in self.data.ch_names})
        self.data.set_montage('standard_1020')

        # butterworth bandpass filter
        self.data.filter(l_freq=1, h_freq=45, method='iir', iir_params=dict(order=8, ftype="butter"))

        # ICA
        self.data.set_eeg_reference('average',projection=True)
        rank = mne.compute_rank(self.data)
        ica = ICA(n_components=rank['eeg'], random_state=97, max_iter="auto", method="fastica")
        ica.fit(self.data)
        #detect eog artifact
        eog_inds, eog_scores = ica.find_bads_eog(self.data, ch_name='Fp1')
        ica.exclude = eog_inds
        raw_clean = ica.apply(self.data.copy())

        data = raw_clean.get_data()
        sfreq = raw_clean.info['sfreq']

        self.processed_data = raw_clean
        
        self.logger.info(f"Data loaded and cleaned. Shape: {self.processed_data.shape}")
        return self.processed_data, sfreq

    def extract_band_features(self) -> pd.DataFrame:
        """Extract frequency band features by region. Returns DataFrame with regional frequency band powers"""
        features = pd.DataFrame()

        for band_name, band_suffix in self.frequency_bands.items():
            for region_name, channels in self.regions.items():
                cols = [col for col in self.data.columns
                        if band_suffix in col
                        and '_Rel' in col
                        and any(ch in col for ch in channels)]
                if cols:
                    # Calculate mean power for this band in this region
                    features[f'{band_name}_{region_name}_mean'] = self.data[cols].mean(axis=1)
                    
            # Calculate overall mean for this band
            band_cols = [col for col in self.data.columns 
                        if band_suffix in col and '_Rel' in col]
            features[f'{band_name}_mean'] = self.data[band_cols].mean(axis=1)
        
        return features
    def create_processed_dataset(self) -> pd.DataFrame:
        """Create the final processed dataset for modeling."""
        self.logger.info("Creating processed dataset...")
        
        # Load and clean data if not already done
        if self.data is None:
            self.clean_data()
            
        # Extract frequency band features
        band_features = self.extract_band_features()
        
        # Combine features with condition
        self.processed_data = pd.concat(
            [band_features, self.data[['Condition']]],
            axis=1
        )
        
        self.logger.info(f"Processed dataset created. Shape: {self.processed_data.shape}")
        return self.processed_data

    def save_processed_data(self, output_path: str = None):
        """Save processed dataset to file."""
        if output_path is None:
            output_path = 'data/processed_data.csv'
            
        if self.processed_data is None:
            self.create_processed_dataset()
            
        self.processed_data.to_csv(output_path, index=False)
        self.logger.info(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocessor = EEGPreprocessor()
    preprocessor.create_processed_dataset()
    preprocessor.save_processed_data()

