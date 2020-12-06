
import pickle
from pathlib import Path, PurePath

from keyword_spotting.feature_extraction.utils import (filter_banks, hamming,
                                                       power_spectrum,
                                                       read_wav)


class PreProcessingPipeline:
    def __init__(self, NNL: int = 512, num_filters: int = 40):
        self.NNL = NNL
        self.num_filter = num_filters

    def extract_features(self, file):
        sample_rate, frames = read_wav(file)
        frames = hamming(frames)
        pow_frames = power_spectrum(frames)
        frames = filter_banks(pow_frames, sample_rate,
                              self.NNL, self.num_filter)
        return frames

    def params(self):
        return self.__dict__

    def hash(self):
        pass

    def filename_ending(self):
        pass

    def transformed_file_name(self, file_name):
        return file_name.split('.')[0] + self.filename_ending() + '.pkl'

    def process(self, name):
        if not isinstance(name, Path):
            name = Path(name)
        file_path_parts = name.parts
        file_name = file_path_parts[-1]
        transformed_file_name = self.transformed_file_name(file_name)
        transformed_file_path = PurePath(
            *file_path_parts[:-1]) / transformed_file_name
        with open(transformed_file_path, 'wb') as file:
            pickle.dump(self.extract_features(name), file)
