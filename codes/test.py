import argparse

from codes.runner.tester import Tester


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', type=str, required=True, help='Path to option file.')
    args = args.parse_args()
    Tester(args.i, 'LR_path')  # might not use this key, depends what Dataset is used.


if __name__ == '__main__':
    main()
