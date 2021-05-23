import argparse
from pathlib import Path
import hashlib
import shutil
import argparse

import numpy as np
import tensorflow as tf
import yaml
from keyword_spotting.feature_extraction.utils import (
    extract_features as keyword_extract_features,
)
from keyword_spotting.feature_extraction.utils import read_wav
from keyword_spotting.model import cnn_inception2, models
import pickle
from sklearn.model_selection import train_test_split
from keyword_spotting.data.utils import distribution_labels


def hash_to_int(s):
    hash_object = hashlib.sha1(str.encode(s))
    return int(hash_object.hexdigest(), 16)

selected_keywords = [
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
]
# fmt: off
folders = [
    "bed", "cat", "dog", "five", "forward", "happy", "learn", "marvin", "one",
    "seven", "six", "tree", "wow", "backward", "bird", "eight", "follow",
    "four", "house", "nine", "sheila", "three", "two", "visual", "zero"
]
# fmt: on


def generate_file_list(dataset_path: Path, labels):
    files = []
    for l in labels[:11]:
        files.extend(list((dataset_path / l).glob("*.wav")))

    files.extend(list((dataset_path / "silence").glob("*.wav")) * 430)

    files = sorted(files, key=lambda path: path.stem)
    return files


def save_files_names(data_path: Path, ds, name):
    files = []
    for x in ds:
        file_parts = x.parts
        file = f"{file_parts[-2]}/{file_parts[-1]}"
        files.append(file)
    DS_PATH = data_path / f"{name}.pickle"
    with open(DS_PATH, "wb") as f:
        pickle.dump(files, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform")
    parser.add_argument(
        "--original-dataset-path",
        type=str,
        required=True,
        help="Folder where the wav are stored",
    )
    parser.add_argument(
        "--dest-dataset-path",
        type=str,
        required=True,
        help="Folder where the wav are stored",
    )
    args = parser.parse_args()
    dataset = Path(args.original_dataset_path)

    output_path = Path(args.dest_dataset_path)
    output_path.mkdir(parents=True, exist_ok=True)

    files = []
    for audio_name in folders:
        files.extend(
            [
                (str(f), hash_to_int(audio_name + "/" + f.name))
                for f in (dataset / audio_name).glob("*.wav")
            ]
        )
    files = sorted(files, key=lambda x: x[1])

    unknown_path = output_path / "unknown"
    unknown_path.mkdir(exist_ok=True, parents=True)
    for f, _ in files[:3200]:
        f = Path(f).resolve()
        shutil.copy(f, unknown_path / f.name)

    for keyword in selected_keywords:
        destination = shutil.copytree(dataset / keyword, output_path / keyword)

    folder = "_background_noise_"
    shutil.copytree(dataset / folder, output_path / folder)

    folder = "silence"
    shutil.copytree(dataset / "_background_noise_", output_path / folder)

    selected_keywords.extend(['unknown', 'silence'])
    
    file_list = generate_file_list(output_path, selected_keywords)
    X_train, X_val_test = train_test_split(file_list, test_size=0.20, random_state=42)
    X_val, X_test = train_test_split(X_val_test, test_size=0.50, random_state=42)

    print(distribution_labels(selected_keywords, X_train))
    print(distribution_labels(selected_keywords, X_test))
    print(distribution_labels(selected_keywords, X_val))

    save_files_names(output_path, X_train, "X_train")
    save_files_names(output_path, X_test, "X_test")
    save_files_names(output_path, X_test, "X_val")
