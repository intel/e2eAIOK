import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='User Provided Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )

    locations = parser.add_argument_group('datasets parameters')

    locations.add_argument('--train_data_pattern', type=str, default='/dataset/tfrecords/train/part*', 
                           help='Pattern of training file names. For example if training files are train_000.tfrecord, '
                                'train_001.tfrecord then --train_data_pattern is train_*')

    locations.add_argument('--eval_data_pattern', type=str, default='/dataset/tfrecords/eval/part*', 
                           help='Pattern of eval file names. For example if eval files are eval_000.tfrecord, '
                                'eval_001.tfrecord then --eval_data_pattern is eval_*')

    locations.add_argument('--dataset_meta_file', type=str, default='data/dataset_meta.yaml',
                           help='Dataset metadata file')

    locations.add_argument('--model_dir', type=str, default='/model/checkpoints',
                           help='Saved model dir for training')

    locations.add_argument('--results_dir', type=str, default='/results',
                           help='Directory to store training results')
                           
    training_params = parser.add_argument_group('training parameters')

    training_params.add_argument('--global_batch_size', type=int, default=1024,
                                 help='Total size of training batch')

    training_params.add_argument('--eval_batch_size', type=int, default=1024,
                                 help='Total size of evaluation batch')

    training_params.add_argument('--num_epochs', type=int, default=1,
                                 help='Number of training epochs')

    training_params.add_argument('--amp', default=False, action='store_true',
                                 help='Enable automatic mixed precision conversion')

    training_params.add_argument('--xla', default=False, action='store_true',
                                 help='Enable XLA conversion')

    training_params.add_argument('--learning_rate', type=float, default=0.001,
                                 help='Learning rate')

    training_params.add_argument('--warmup_steps', type=float, default=1000,
                                 help='Number of learning rate warmup steps')

    training_params.add_argument('--metric', type=str, default='AUC', help='Evaluation metric')

    training_params.add_argument('--metric_threshold', type=float, default=0, help='Metric threshold used for training early stop')

    training_params.add_argument('--optimizer', type=str, default='SGD', help='Training optimizer')

    training_params.add_argument('--loss', type=str, default='BinaryCrossentropy', help='Training loss function')

    run_params = parser.add_argument_group('run mode parameters')

    run_params.add_argument('--evaluate', default=False, action='store_true',
                            help='Only perform an evaluation on the validation dataset, don\'t train')

    run_params.add_argument('--platform', type=str, default='tensorflow',
                            help='Training framework')

    return parser.parse_args()