from runs import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("Eager execution: {}".format(tf.executing_eagerly()))

if __name__ == "__main__":
    parameters = {
        "devices": ['GPU', 'CPU'],
        "model": ['simpleNet', 'sequentialNet'],
        "epochs": [1]
    }

    run_all(parameters)
