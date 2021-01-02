import argparse

from codes.runner.tester import Tester


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', type=str, required=True, help='Path to option file.')
    args = args.parse_args()
    Tester(
        args.i,
        # TODO: Might not use these keys, depends what Dataset is used.
        #       These keys are being used, as that's what was used in the old test.py
        path_key='LR_path',
        wanted_visuals=['SR'],
        metric_visuals=('SR', 'HR')
    )


if __name__ == '__main__':
    main()
