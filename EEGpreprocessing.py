import pandas as pd
import numpy as np
import logging
import mne
from mne.preprocessing import ICA
import os

class EEGPreprocessor:
    """
    Class for pre-processing EEG data for identifying frequency band features of chronic pain.
    Extracts frequency band power information from raw EEG data for AD analysis
    """
    def __init__(self, data_path: str = '/Users/arielmotsenyat/Documents/coding-workspace/mun_pain_data'):
        """initialize preprocessor with correct data path"""
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.logging_setup()

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
    
    def logging_setup(self):
        """setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self) -> pd.DataFrame:
        """this function will load, filter, apply ICA, and return cleaned EEG data"""
        self.logger.info("Loading data...")
        self.data = mne.io.read_raw_brainvision(self.data_path, preload=True)

        self.data.set_channel_types({ch: "eeg" for ch in self.data.ch_names if ch not in ['LE', 'RE', 'Ne', 'Ma']}) # drop non-EEG channels
        self.data.drop_channels(['LE', 'RE', 'Ne', 'Ma'])
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

        self.processed_data = raw_clean
        
        self.logger.info(f"Data loaded and cleaned. Shape: {self.processed_data.get_data().shape}")
        return self.processed_data 

if __name__ == "__main__":
    data_path = '/Users/arielmotsenyat/Documents/coding-workspace/mun_pain_data'
    clean_data_path = ('/Users/arielmotsenyat/Documents/coding-workspace/ChronicPainClassifier/clean_data')
    os.makedirs(clean_data_path)

    for sub in range(1,48):
        sub_id = f"sub-{sub:02d}"
    
        hc_file = os.path.join(data_path, f"sub-NCCPhc{sub:02d}", 'eeg', f"sub-NCCPhc{sub:02d}_task-closed_eeg.vhdr")
        pa_file = os.path.join(data_path, f"sub-NCCPpa{sub:02d}", 'eeg', f"sub-NCCPpa{sub:02d}_task-closed_eeg.vhdr")
    
        #skip participant if any file is missing
        if not (os.path.exists(hc_file) and os.path.exists(pa_file)):
            print(f"Skipping {sub_id}: missing healthy control or patient file")
            continue
    
        for condition, file_path in zip(['hc', 'pa'], [hc_file, pa_file]):
            if condition == 'hc':
                preprocessor = EEGPreprocessor(hc_file)
                cleaned_raw_hc = preprocessor.clean_data()
                output_filename = f"sub-NCCP{condition}{sub:02d}_task-closed_clean-raw.fif"
                output_path = os.path.join(clean_data_path, output_filename)
                cleaned_raw_hc.save(output_path, overwrite=True)
                
            elif condition == 'pa':
                preprocessor = EEGPreprocessor(pa_file)
                cleaned_raw_pa = preprocessor.clean_data()
                output_filename = f"sub-NCCP{condition}{sub:02d}_task-closed_clean-raw.fif"
                output_path = os.path.join(clean_data_path, output_filename)
                cleaned_raw_pa.save(output_path, overwrite=True)

