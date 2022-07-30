import tensorflow as tf
import itertools
from time import time
from models import simpleNet, sequentialNet
from input_pipeline import input_pipeline


def run_one(device, model, epochs):
    # Load & create mnist dataset
    train_ds, test_ds = input_pipeline()

    # Model choice
    if model == 'simpleNet':
        model = simpleNet()
    elif model == 'sequentialNet':
        model = sequentialNet()

    # Device choice
    if device == 'CPU':
        with tf.device("CPU:0"):
            model.compile()
            model.fit(epochs, train_ds, test_ds, verbose=1)

    elif device == 'GPU':
        with tf.device("GPU:0"):
            model.compile()
            model.fit(epochs, train_ds, test_ds, verbose=1)


def run_all(parameters):
    keys, values = zip(*parameters.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for config in configs:
        start = time()
        run_one(*config.values())
        end = time()
        print(config, '->', f'{round(end-start)}s')
