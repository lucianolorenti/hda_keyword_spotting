import argparse
from pathlib import Path
import hashlib
import shutil


def hash_to_int(s):
    hash_object = hashlib.sha1(str.encode(s))
    return int(hash_object.hexdigest(), 16)


selected_keywords = [
    'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'
]

folders = [
    'bed', 'cat', 'dog', 'five', 'forward', 'happy', 'learn', 'marvin', 'one',
    'seven', 'six', 'tree', 'wow', 'backward', 'bird', 'eight', 'follow',
    'four', 'house', 'nine', 'sheila', 'three', 'two', 'visual', 'zero'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform')
    parser.add_argument('--original-dataset-path',
                        type=str,
                        required=True,
                        help='Folder where the wav are stored')
    parser.add_argument('--dest-dataset-path',
                        type=str,
                        required=True,
                        help='Folder where the wav are stored')
    args = parser.parse_args()
    dataset = Path(args.original_dataset_path)

    output_path = Path(args.dest_dataset_path)

    folders = [
        'bed', 'cat', 'dog', 'five', 'forward', 'happy', 'learn', 'marvin',
        'one', 'seven', 'six', 'tree', 'wow', 'backward', 'bird', 'eight',
        'follow', 'four', 'house', 'nine', 'sheila', 'three', 'two', 'visual',
        'zero'
    ]

    dataset = Path('/home/luciano/speech')
    files = []
    for audio_name in folders:
        files.extend([(str(f), hash_to_int(audio_name + '/' + f.name))
                      for f in (dataset / audio_name).glob("*.wav")])
    files = sorted(files, key=lambda x: x[1])

    unknown_path = output_path / 'unknown'
    unknown_path.mkdir(exist_ok=True, parents=True)
    for f, _ in files[:3200]:
        f = Path(f).resolve()
        shutil.copy(f, unknown_path / f.name)
