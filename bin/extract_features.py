import argparse
import logging
import pickle
from pathlib import Path, PurePath
from typing import Union

import numpy as np
import tensorflow as tf
from keyword_spotting.data import Dataset
from keyword_spotting.feature_extraction.extractor import FeatureExtractor
from tqdm.auto import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('feature_extraction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform')
    parser.add_argument('--input-path',
                        type=str,
                        required=True,
                        help='Folder where the wav are stored')
    parser.add_argument('--output-path',
                        type=str,
                        required=True,
                        help='Ouput folder')

    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='Name of the dataset')

    args = parser.parse_args()

    path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    dataset = Dataset(path)
    extractor = FeatureExtractor(dataset, output_path, args.name)
    extractor.write()
