import sys
import os

from trainer.utils.arguments import parse_args

def main():
    args = parse_args()
    if args.platform == 'tensorflow':
        from tf.setup import create_config
        from tf.model_loader import ModelLoader
        from tf.run import train, evaluate
    elif args.platform == 'pytorch':
        from pt.setup import create_config
        from pt.model_loader import ModelLoader
        from pt.run import train, evaluate
    config = create_config(args)
    model = ModelLoader(args.model_dir, args.platform).load_model()

    if args.evaluate:
        evaluate(args, model, config)
    else:
        metric = train(args, model, config)
        file = os.path.join(args.results_dir, 'metric.txt')
        with open(file, 'w') as f:
            f.writelines(str(metric))


if __name__ == '__main__':
    main()