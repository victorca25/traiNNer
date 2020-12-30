import argparse

from codes.runner.trainer import Trainer


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', type=str, required=True, help='Path to option file.')
    args = args.parse_args()
    Trainer(args.i)


if __name__ == '__main__':
    main()
