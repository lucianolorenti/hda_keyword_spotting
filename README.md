# Installation


## Download the dataset
```bash
wget dataset_url
tar -zxvf dataset_url
```

## Installation

```bash
pip install -e .
```

# Dataset creation
The dataset creation will select the following keywords for building the dataset:

* down
* go
* left
* no
* off
* on
* right
* stop 
* up 
* yes

And will create the unknown folder composed by a sample from the remaining words present in the dataset.

```bash
python bin/build_dataset.py  \
    --original-dataset-path INPUT_DATASET_PATH \
    --dest-dataset-path OUTPUT_DATASET_PATH
```    


The background noise folder will be use for the silence label, and for doing data augmentation on the audios.