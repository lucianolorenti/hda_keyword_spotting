import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from time import time

import yaml
from keyword_spotting.model import cnn_inception2, models
from keyword_spotting.predictions import (
    evaluate_perdictions,
    labels,
    labels_dict,
    predictions_per_song,
)
from keyword_spotting.train import PerAudioAccuracy, build_dataset_generator, load_data
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logging.basicConfig()
logger = logging.getLogger("hda")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )

    args = parser.parse_args()

    config = None
    with open(args.config, "r") as file:
        config = yaml.load(file.read(), Loader=yaml.SafeLoader)

    data_path = Path(args.dataset)
    output_path = Path(args.output_dir)

    X_train, X_val, X_test = load_data(data_path)

    ds_train = build_dataset_generator(
        X_train, data_path, config["model"]["windowed"], noise=True, shuffle=True
    )

    number_of_classes = len(labels)

    input_shape = [100, 40] if not config["model"]["windowed"] else [40, 40]
    params = config["model"].get("params", {})

    model = models[config["model"]["name"]](input_shape, number_of_classes, **params)
    model.summary()
    epochs = config["train"]["epochs"]

    output_filename = (
        config["model"]["name"] + "_" + datetime.now().strftime("%H_%M_%S_%b_%d_%Y")
    )

    model_path = output_path / output_filename
    results_path = output_path / (output_filename + "_results.pkl")
    print(model_path, results_path)
    batch_size = config["train"]["batch_size"]

    callbacks = [EarlyStopping(patience=5)]

    if config["train"]["reduce_on_plateau"]:
        callbacks.append(ReduceLROnPlateau(patience=2, verbose=1, min_lr=0.00001))

    if config["model"]["windowed"]:
        callbacks.append(PerAudioAccuracy(model, X_val))
    ds_val = build_dataset_generator(
        X_val[:5], data_path, config["model"]["windowed"], noise=False, shuffle=False
    )

    start = time()
    history = model.fit(
        ds_train.batch(batch_size),
        validation_data=ds_val.batch(batch_size),
        epochs=epochs,
        # steps_per_epoch=asd // batch_size,
        callbacks=callbacks,
    )
    total_time = time() - start
    model.save_weights(str(model_path))

    if config["model"]["windowed"]:
        results = predictions_per_song(model, X_test)
    else:
        ds_test = build_dataset_generator(
            X_test[:5], data_path, config["model"]["windowed"], noise=False, shuffle=False
        )
        results = model.predict(ds_test.batch(128))

    with open(results_path, "wb") as file:
        pickle.dump((config, total_time, results), file)
