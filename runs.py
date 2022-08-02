import tensorflow as tf
import itertools
from time import time
from models import *
from input_pipeline import input_pipeline


def run_one(device, model, prefetch, epochs, verbose):
    # Load & create mnist dataset
    train_ds, test_ds = input_pipeline(prefetch)

    # Model choice
    if model == 'simpleNet':
        model = simpleNet()
    elif model == 'sequentialNet':
        model = sequentialNet()
    elif model == 'simpleNet_graph':
        model = simpleNet_graph()

    # Device choice
    if device == 'CPU':
        with tf.device("CPU:0"):
            model.compile()
            model.fit(epochs, train_ds, test_ds, verbose=verbose)

    elif device == 'GPU':
        with tf.device("GPU:0"):
            model.compile()
            model.fit(epochs, train_ds, test_ds, verbose=verbose)


def run_all(parameters):
    keys, values = zip(*parameters.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for config in configs:
        start = time()
        run_one(*config.values())
        end = time()
        print(config, '->', f'{round(end-start)}s')
