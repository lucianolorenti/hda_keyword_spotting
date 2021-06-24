import argparse
import logging
import pickle
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from keyword_spotting.model import PatchEncoder, Patches, models
from keyword_spotting.predictions import evaluate_predictions, labels
from keyword_spotting.train import build_dataset_generator, load_data
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)

logging.basicConfig()
logger = logging.getLogger("hda")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--model-path", type=str, required=True, help="Model-path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")

    args = parser.parse_args()
    data_path = Path(args.dataset)
    config_file = args.model_path + "_results.pkl"
    with open(config_file, "rb") as f:
        config, train_time, predictions = pickle.load(f)
    print(config)
    input_shape = [100, 40]
    number_of_classes = 12

    model_path = args.model_path
    h5_file = Path(model_path + ".h5")
    if h5_file.is_file():
        model_path = str(h5_file)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder},
    )

    # model = models[config["model"]["name"]](input_shape, number_of_classes, **config["model"]["params"])
    # model.load_weights(args.model_path)
    model.summary()

    _, _, X_test = load_data(data_path)
    ds_test = build_dataset_generator(
        X_test, data_path, config["model"]["windowed"], noise=False, shuffle=False
    )
    start = time()
    y_pred = model.predict(ds_test.batch(128))
    total_time = time() - start
    y_true, y_pred = evaluate_predictions(y_pred, data_path)
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Train Time: {train_time}")
    print(f"Test Time: {total_time}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(11, 11))
    disp.plot(ax=ax)
    fig.tight_layout()
    fig.savefig('cm.png')
