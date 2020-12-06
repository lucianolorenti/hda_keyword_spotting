
import pickle
from pathlib import Path, PurePath

from keyword_spotting.feature_extraction.utils import (extract_features,
                                                       filter_banks, hamming,
                                                       power_spectrum,
                                                       read_wav)


class Transformer:
    def transform(self, X):
        return extract_features(X)
