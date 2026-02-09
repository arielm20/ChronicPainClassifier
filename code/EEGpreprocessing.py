
import pandas as pd
import numpy as np
import logging
import mne
from mne.preprocessing import ICA
import os

class EEGPreprocessor:
    """
    Class for applying filter and ICA on raw EEG data (brain vision format), to then extract frequency band power information for chronic pain analysis.
    """
    def __init__(self, data_path = '/Users/arielmotsenyat/Documents/coding-workspace/classifier_data/mun_pain_data'):
        """initialize preprocessor with correct data path"""
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.logging_setup()

        self.frequency_bands = {
            'Delta': (1, 3),
            'Theta': (3, 7),
            'Alpha': (7, 13),
            'Beta': (13, 30)
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
        """this function will load, filter, apply ICA, and return processed EEG data"""
        self.logger.info("Loading data...")
        self.data = mne.io.read_raw_brainvision(self.data_path, preload=True)

        self.data.set_channel_types({ch: "eeg" for ch in self.data.ch_names if ch not in ['LE', 'RE', 'Ne', 'Ma']}) # drop non-EEG channels
        self.data.drop_channels(['LE', 'RE']) # drop the left and right ear channels
        self.data.set_montage('standard_1020')

        # butterworth bandpass filter 1-45 Hz
        self.data.filter(l_freq=1, h_freq=45, method='iir', iir_params=dict(order=8, ftype="butter"))

        # ICA
        self.data.set_eeg_reference('average',projection=True)
        rank = mne.compute_rank(self.data)
        ica = ICA(n_components=rank['eeg'], random_state=97, max_iter="auto", method="fastica")
        ica.fit(self.data)

        # detect eog artifact
        eog_inds, eog_scores = ica.find_bads_eog(self.data, ch_name='Fp1')
        ica.exclude = eog_inds
        self.processed_data = ica.apply(self.data.copy())
        
        self.logger.info(f"Data loaded and cleaned. Shape: {self.processed_data.get_data().shape}")
        return self.processed_data

    def extract_band_features(self) -> pd.DataFrame:
        """Extract relative frequency band power by region. Returns DataFrame with relative band powers by region"""
        self.logger.info("Extracting frequency band features...")

        features_dict = {}
        band_regional_power = {}
        band_total_power = {}
        regional_total_powers = {}

        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            self.logger.info(f"Processing {band_name} band ({low_freq}-{high_freq} Hz)...")

            # filter data for this band
            filtered = self.processed_data.copy().filter(low_freq, high_freq, fir_design='firwin', verbose=False)
            # get data array shape
            data = filtered.get_data()
            # calculate power for each channel
            power = np.mean(data**2, axis=1)
            band_total_power[band_name] = np.mean(power)
            # create dictionary for each channel and power values
            channel_powers = {ch: power[idx] for idx, ch in enumerate(self.processed_data.ch_names)}

            # calculate mean power for each brain region
            for region_name, channels in self.regions.items():
                # find channels that exist in both the region and the data
                available_channels = [ch for ch in channels if ch in self.processed_data.ch_names]

                if available_channels:
                    region_powers = [channel_powers[ch] for ch in available_channels]
                    region_power_mean = np.mean(region_powers)

                    if region_name not in band_regional_power:
                        band_regional_power[region_name] = {}
                    band_regional_power[region_name][band_name] = region_power_mean
                else:
                    self.logger.warning(f"No channels found for {region_name} region")
            
            # calculate relative power for each region
            for region_name in self.regions.keys():
                regional_total_powers[region_name] = sum(band_regional_power.get(region_name, {}).values())
                for band_name in self.frequency_bands.keys():
                    if region_name in band_regional_power and band_name in band_regional_power[region_name]:
                        absolute_power = band_regional_power[region_name][band_name]
                        regional_total = regional_total_powers[region_name]
                    features_dict[f'{band_name}_{region_name}_rel'] = absolute_power / (regional_total+1e-10)

            # calculate total relative power 
            total_power_all_bands = sum(band_total_power.values())
            for band_name, band_power in band_total_power.items():
                features_dict[f'{band_name}_total_rel'] = band_power / (total_power_all_bands+1e-10)

            # calculate theta-alpha ratio
            if 'Theta' in band_total_power and 'Alpha' in band_total_power:
                features_dict['theta_alpha_ratio'] = (features_dict['Theta_total_rel'] / features_dict['Alpha_total_rel'])
            else:
                continue
        
        features_df = pd.DataFrame([features_dict])
        
        self.logger.info(f"Extracted {len(features_dict)} band features")
        return features_df

    def create_processed_dataset(self, condition: str = None) -> pd.DataFrame:
        """Create the final processed dataset for modeling.
        Args:
            condition: Optionaly include condition label 'cbp' for chronic back pain, 'fm' for fibromyalgia, and 'nccp for mixed chronic pain conditions
        """
        self.logger.info("Creating processed dataset...")
        
        # load and clean data if not already done
        if self.processed_data is None:
            self.clean_data()
            
        # extract frequency band features
        band_features = self.extract_band_features()

        # add condition if provided
        if condition is not None:
            band_features['Condition'] = condition
       
        self.processed_data_df = band_features
        self.processed_data_df['Participant #'] = sub_id
        self.logger.info(f"Processed dataset created. Shape: {self.processed_data_df.shape}")
        return self.processed_data_df        
    
    def save_processed_data(self, output_path: str = None):
        """Save processed dataset to file."""
        if output_path is None:
            output_path = '/Users/arielmotsenyat/Documents/coding-workspace/ChronicPainClassifier/data/processed_data_rel_all_pain.csv'
            
        if self.processed_data_df is None:
            self.create_processed_dataset()
        
        file_exists = os.path.exists(output_path)
        self.processed_data_df.to_csv(output_path, mode='a', header=(not file_exists), index=False)
        self.logger.info(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    data_path = '/Users/arielmotsenyat/Documents/coding-workspace/classifier_data/mun_pain_data'
    for sub in range(1,47):
        sub_id = f"sub-{sub:02d}"
    
        cbp_file = os.path.join(data_path, f"sub-CBPpa{sub:02d}", 'eeg', f"sub-CBPpa{sub:02d}_task-closed_eeg.vhdr")
        fm_file = os.path.join(data_path, f"sub-FMpa{sub:02d}", 'eeg', f"sub-FMpa{sub:02d}_task-closed_eeg.vhdr")
        nccp_file = os.path.join(data_path, f"sub-NCCPpa{sub:02d}", 'eeg', f"sub-NCCPpa{sub:02d}_task-closed_eeg.vhdr")
    
        for condition, file_path in zip(['cbp', 'fm', 'nccp'], [cbp_file, fm_file, nccp_file]):
            if not(os.path.exists(file_path)):
                print(f"Skipping {sub_id} {condition}: missing patient file")
                continue
            
            if condition == 'cbp':
                preprocessor = EEGPreprocessor(cbp_file)
                preprocessor.create_processed_dataset(condition=condition)
                
            elif condition == 'fm':
                preprocessor = EEGPreprocessor(fm_file)
                preprocessor.create_processed_dataset(condition=condition)
            
            elif condition == 'nccp':
                preprocessor = EEGPreprocessor(nccp_file)
                preprocessor.create_processed_dataset(condition=condition)
            preprocessor.save_processed_data()
