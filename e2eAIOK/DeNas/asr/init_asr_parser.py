import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description="Transformer Training Reference")
    parser.add_argument(
        "--param_file", type=str,
        help="A yaml-formatted file using the extended YAML syntax. "
        "defined by SpeechBrain.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="The device to run the experiment on (e.g. 'cuda:0')",
    )
    parser.add_argument("--local_rank", type=int, help="Rank on local machine")
    parser.add_argument(
        "--distributed_launch", type=bool, default=False,
        help="This flag enables training with DDP. Assumes script run with "
        "`torch.distributed.launch`",
    )
    parser.add_argument(
        "--distributed_backend", type=str, default="gloo",
        help="One of {ccl, gloo, mpi}",
    )
    parser.add_argument(
        "--find_unused_parameters", default=False, action="store_true",
        help="This flag disable unused parameters detection",
    )
    parser.add_argument(
        "--max_grad_norm", type=float,
        help="Gradient norm will be clipped to this value, "
        "enter negative value to disable.",
    )
    parser.add_argument(
        "--grad_accumulation_factor", type=int,
        help="Number of batches to accumulate gradients before optimizer step",
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Set global seed"
    )
    parser.add_argument(
        "--output_folder", type=str,
        help="Define output folder"
    )
    parser.add_argument(
        "--save_folder", type=str,
        help="Checkpoint save folder"
    )
    parser.add_argument(
        "--data_folder", type=str,
        help="Dataset folder"
    )
    parser.add_argument(
        "--train_csv", type=str,
        help="Train meta data file"
    )
    parser.add_argument(
        "--valid_csv", type=str,
        help="Evaluation meta data file"
    )
    parser.add_argument(
        "--test_csv", type=str, nargs="+",
        help="Test meta data file"
    )
    parser.add_argument(
        "--skip_prep", type=bool,
        help="Skip preprocess"
    )
    parser.add_argument(
        "--lm_model_ckpt", type=str,
        help="LM checkpoint file"
    )
    parser.add_argument(
        "--tokenizer_ckpt", type=str,
        help="Tokenizer checkpoint file"
    )

    return parser.parse_args(args)
