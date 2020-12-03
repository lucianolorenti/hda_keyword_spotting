import argparse
import logging
import pickle
from pathlib import Path, PurePath

from keyword_spotting.feature_extraction.utils import extract_features
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('feature_extraction')

# if __name__ == '__main__':
path = Path('/home/lucianolorenti/data')
temp = {}

for name in tqdm(list(path.glob('**/*.wav'))):
    file_path_parts = name.parts
    file_name = file_path_parts[-1]
    transformed_file_name = file_name.split('.')[0] + '.pkl'
    transformed_file_path = PurePath(
        *file_path_parts[:-1]) / transformed_file_name
    with open(transformed_file_path, 'wb') as file:
        pickle.dump(extract_features(name), file)
