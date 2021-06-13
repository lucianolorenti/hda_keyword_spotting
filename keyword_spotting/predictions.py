from pathlib import Path
import numpy as np
from scipy import stats
from keyword_spotting.feature_extraction.utils import extract_features, read_wav, windowed

labels = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
    "silence",
]
labels_dict = {l: i for i, l in enumerate(labels)}

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / (periods)
    return np.convolve(data_set, weights, mode='valid')


def moving_average_1(data, w):
    b = np.abs(np.random.randn(50)*np.random.rand(50)*500)
    b = b / np.sum(b)
    return [np.mean(b[max(0, j-w):j]) for j in range(b.shape[0])]

def posterior_smoothing(probas, w):
    """Computes the average posterior for a any frame j considering the posterior probabilities of the previous frames. 
    The numbers of probabilities to smooth is taking account the size of the parameter w, the size of the smoothing window.  
    The idea is to smooth the noisy of the posteriors.  

    Args:
        probas (numpy.ndarray(float)): arrays of posteriors for a specific label per any frame. 
        w (int): size of the smoothing window

    Returns:
        numpy.ndarray(float): An array of smoothed posteriors per each frame.
    """    
    return [np.mean(probas[max(0, j-w):j]) for j in range(1, probas.shape[0]+1)]

def confidence(probas, w):
    """Computes a confidence value per every frame taking account all their smooth probalities in a sliding window w. 
    The method multiplies all the maximum smooth probabilities for every label and finally computes the n root of that value. 

    Args:
        probas (numpy.ndarray(umpy.ndarray(float))): arrays of smooth posteriors for each label per any frame. 
        w (int): size of the sliding window

    Returns:
        (numpy.ndarray(float)): An array of confidende values for each frame.
    """    
    labels = probas.shape[1]
    print(probas.shape)
    confidences = []
    for j in range(1, probas.shape[0]+1):
        hmax = max(0, j-w)
        prod = np.prod([np.max(probas[hmax:j, i]) for i in range(0, labels)])
        confidences.append(prod**(1/labels))
    return confidences
    

def get_predictions(labels_posteriors, total_labels, smooth_w=2, max_w=5):
    """ Get the prediction for a wav throught the posterior handling of the frames.
    See paper 

    Args:
        labels_posteriors (numpy.ndarray(numpy.ndarray(float))): arrays of posteriors for each label per any frame.
        total_labels (int): number of labels
        smooth_w (int, optional): Size of the smoothing window. Defaults to 2.
        max_w (int, optional): Size of the maximun sliding window size. Defaults to 5.

    Returns:
        int: Label predicted.
    """    
    smooth_posteriors = np.zeros(labels_posteriors.shape)

    for label in range(total_labels):
        posteriors_label=labels_posteriors[:, label]
        smooth_posteriors[:, label] = posterior_smoothing(posteriors_label, smooth_w)
    
    confidences = confidence(smooth_posteriors, max_w )
    
    # Get the frame with the highest probability
    
    frame_best_confidence = np.argmax(confidences)

    # For the frame with highest confidence we obtain the label with the highest posterior
    return np.argmax(labels_posteriors[frame_best_confidence])


def predictions_per_song(model, dataset):
    results = []
    for audio_file in dataset:
        try:
            label = Path(audio_file).resolve().parts[-2]
            label = labels_dict[label]
            sample_rate, signal = read_wav(audio_file)
            data = extract_features(signal, sample_rate)
            data, labels = windowed(data, label)
            predicted = model.predict(data)
            results.append((audio_file, label, predicted))
        except:
            pass

    return results


def evaluate_perdictions(predictions):
    preds = []
    trues = []
    for _, real_label, labels_posteriors in predictions:
        preds.extend((np.ones(labels_posteriors.shape[0])*real_label).astype('int'))
        trues.extend(np.argmax(labels_posteriors,axis=1))    
    return preds, trues