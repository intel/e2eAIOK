import argparse


def init_cnn_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--best_model_structure', default='./best_model_structure.txt', type=str, metavar='BMS',
                        help='Best model structure string')
    parser.add_argument('--num-classes', type=int, default=10, help='how to prune')
    parser.add_argument("--dist-backend", type=str, default="ccl")
    parser.add_argument('--train-batch-size', default=128, type=int)
    parser.add_argument('--eval-batch-size', default=128, type=int)

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='CIFAR10', choices=["CIFAR10","CIFAR100"],
                        type=str, help='Image Net dataset path')


    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser