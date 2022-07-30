import argparse


def namespace():
    parser = argparse.ArgumentParser(description='Performance choice')

    parser.add_argument('--device', type=str, default='GPU')
    parser.add_argument('--model', type=str, default='simpleNet')

    return parser.parse_args()
