

from keyword_spotting.data import Dataset
from keyword_spotting.model import get_model_2

dataset = Dataset('/home/luciano/speech')
dataset.to_numpy()
model = get_model_2(dataset.shape, dataset.number_of_classes)

train_data, val_data, test_data = dataset.get_sequences(batch_size=256)
model.fit(train_data,
          validation_data=val_data, epochs=15)
