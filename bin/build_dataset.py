import argparse
from pathlib import Path
import hashlib
import shutil


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
