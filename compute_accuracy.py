import argparse
from utils.eval_outputs import eval_gsm8k_or_aqua_outputs, eval_mgsm_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='dataset to compute',
        choices=['mgsm', 'aqua', 'gsm8k', 'gsm-plus-digits'])
    parser.add_argument('path', help='path to the output json file')
    args = parser.parse_args()
    
    if args.data == 'mgsm':
        eval_mgsm_outputs(args.path, verbose=True)
    else:
        eval_gsm8k_or_aqua_outputs(args.data, args.path, verbose=True)


if __name__ == '__main__':
    main()
