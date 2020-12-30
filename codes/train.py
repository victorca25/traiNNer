import argparse

from codes.scripts.classes.trainer import Trainer


def main():
    # arguments
    args = argparse.ArgumentParser()
    args.add_argument('-i', type=str, required=True, help='Path to option JSON file.')
    args = args.parse_args()

    trainer = Trainer(args.i)


if __name__ == '__main__':
    main()
