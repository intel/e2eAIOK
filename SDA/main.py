import argparse
import initsda

from SDA.SDA import SDA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DLRM', help='Define model')
    parser.add_argument('--dataset_meta_path', type=str, required=True, help='Dataset metadata file')
    parser.add_argument('--train_path', type=str, help='Train dataset path')
    parser.add_argument('--eval_path', type=str, help='Evaluation dataset path')
    return parser.parse_known_args()

if __name__ == '__main__':
    sda_args, model_args = parse_args()

    sda = SDA(sda_args.model, sda_args.dataset_meta_path, sda_args.train_path, sda_args.eval_path, model_args)
    sda.launch()