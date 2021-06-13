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

# Fitting the model
The model are fitted with the bin/fit.py. A configuration file must be provided to the program. The configuration file must 
contain

Model:
    Name: The name of the model: Possible values are
    Params: An object with the parameters needed to construct the model specified
    WIndowed: Wether to use the model for windowed predictions or using the whole audio as input

Train:
    Batch_size:
    epochs:
    reduce_on_plateau: 

```yaml
model:
  name: 'cnn_residual2'
  params:
    n_residuals: 3
    n_filters: 19
    learning_rate: 0.01
  windowed: True
    

train:
  batch_size: 8
  epochs: 2
  reduce_on_plateau: True
```
