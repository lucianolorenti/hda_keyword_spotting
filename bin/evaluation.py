import argparse
from keyword_spotting.train import build_dataset_generator, load_data
import logging
import pickle

from pathlib import Path

from keyword_spotting.predictions import evaluate_predictions
from sklearn.metrics import accuracy_score
import yaml
from keyword_spotting.model import  models


logging.basicConfig()
logger = logging.getLogger("hda")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--model-path", type=str, required=True, help="Model-path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    

    args = parser.parse_args()
    data_path = Path(args.dataset)
    config_file = args.model_path + '_results.pkl'
    with open(config_file, 'rb') as f:
        config, time, predictions = pickle.load(f)
    print(config)
    input_shape = [100, 40]
    number_of_classes=12    

    model = models[config["model"]["name"]](input_shape, number_of_classes, **config["model"]["params"])
    model.summary()

    _, _, X_test = load_data(data_path)
    ds_test = build_dataset_generator(
        X_test, data_path, config["model"]["windowed"], noise=False, shuffle=False
    )
    y_pred = model.predict(ds_test.batch(128))
    y_true, y_pred = evaluate_predictions(y_pred, Path('/home/luciano/speech_2'))
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print(f'Time: {time}')