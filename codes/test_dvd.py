import argparse

from codes.runner.tester import Tester


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', type=str, required=True, help='Path to option file.')
    args = args.parse_args()
    Tester(
        args.i,
        path_key='in_path',
        wanted_visuals=['top_fake', 'bottom_fake'],
        metric_visuals=('top_fake', 'top_real')
    )


if __name__ == '__main__':
    main()
