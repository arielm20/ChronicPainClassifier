import pandas as pd
import mne

class EEGPreprocessor:
    """
    Class for pre processing EEG data for identifying frequency band features of chronic pain.
    Extracts frequency information from raw EEG data for AD analysis
    """