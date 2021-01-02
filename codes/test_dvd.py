import argparse

from codes.runner.tester import Tester


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', type=str, required=True, help='Path to option file.')
    args = args.parse_args()
    Tester(args.i, 'in_path')


if __name__ == '__main__':
    main()
