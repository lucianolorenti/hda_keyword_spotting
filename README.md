# Installation


## Download the dataset
```bash
mkdir dataset
cd dataset
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -zxf speech_commands_v0.02.tar.gz
```

## Installation

```bash
pip install -e hda_keyword_spotting/
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
The model are fitted with the `bin/train.py`. A configuration file must be provided to the program. The configuration file must 
contain

Model:
    Name: The name of the model: Possible values are
    Params: An object with the parameters needed to construct the model specified
    Windowed: Wether to use the model for windowed predictions or using the whole audio as input

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
```bash
python bin/train.py --config bin/train_config_vit.yml --output-dir OUTPUT_DIR --dataset DATASET_PATH
```    


# Evaluating the model
```bash
python bin/evaluation.py --model-path models/res3 --dataset DATASET_PATH
```