import pandas as pd
import matplotlib as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os
import mne
from mne_connectivity import spectral_connectivity_epochs
import scipy as sp
from scipy import signal

class feature_analysis:
    """A class to explore EEG data from the datasent focused on Chronic Pain analysis"""

    def __init__(self, raw: mne.io.Raw, data_path: str = '/Users/arielmotsenyat/Documents/coding-workspace/mun_pain_data'):
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

    def _load_Data(self, data_path: str) -> pd.DataFrame:
        """Load EEG data and perform initial validation"""
        
