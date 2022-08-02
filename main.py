from runs import *
import os
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("Eager execution: {}".format(tf.executing_eagerly()))

if __name__ == "__main__":
    parameters = {
        "devices": ['GPU', 'CPU'],
        "model": ['simpleNet', 'sequentialNet', 'simpleNet_graph'],
        "prefetch": [True, False]
    }

    # run_all(parameters)
    start = time()
    run_one('GPU', 'simpleNet', False,  epochs=2, verbose=0)
    end = time()
    print(f'Execution time : {end-start}')

    start = time()
    run_one('GPU', 'simpleNet_graph', False,  epochs=2, verbose=0)
    end = time()
    print(f'Execution time : {end-start}')
