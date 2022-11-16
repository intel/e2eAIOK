import torch
import torch.distributed as dist
import logging
import os
import argparse


logger = logging.getLogger(__name__)


def create_experiment_directory(experiment_directory):
    try:
        # all writing command must be done with the main_process
        if is_main_process():
            if not os.path.isdir(experiment_directory):
                os.makedirs(experiment_directory)

            # Log beginning of experiment!
            logger.info("Beginning experiment!")
            logger.info(f"Experiment folder: {experiment_directory}")

    finally:
        # wait for main_process if ddp is used
        ddp_barrier()

def init_log(distributed_launch):
    if distributed_launch:
        if dist.get_rank() == 0:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

def run_on_main(
    func,
    args=None,
    kwargs=None,
    post_func=None,
    post_args=None,
    post_kwargs=None,
    run_post_on_main=False,
):
    # Handle the mutable data types' default args:
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if post_args is None:
        post_args = []
    if post_kwargs is None:
        post_kwargs = {}

    if is_main_process():
        # Main comes here
        try:
            func(*args, **kwargs)
        finally:
            ddp_barrier()
    else:
        # Others go here
        ddp_barrier()
    if post_func is not None:
        if run_post_on_main:
            # Just run on every process without any barrier.
            post_func(*post_args, **post_kwargs)
        elif not is_main_process():
            # Others go here
            try:
                post_func(*post_args, **post_kwargs)
            finally:
                ddp_barrier()
        else:
            # But main comes here
            ddp_barrier()

def is_main_process():
    if not dist.is_initialized() or dist.get_rank() == 0:
        return True
    else:
        return False

def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()

def check_gradients(modules, loss, max_grad_norm, nonfinite_count, nonfinite_patience=3):
    if not torch.isfinite(loss):
        nonfinite_count += 1
        # Print helpful debug info
        logger.warn(f"Loss is {loss}.")
        for p in modules.parameters():
            if not torch.isfinite(p).all():
                logger.warn("Parameter is not finite: " + str(p))
        # Check if patience is exhausted
        if nonfinite_count > nonfinite_patience:
            raise ValueError(
                "Loss is not finite and patience is exhausted. "
                "To debug, wrap `fit()` with "
                "autograd's `detect_anomaly()`, e.g.\n\nwith "
                "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
            )
        else:
            logger.warn("Patience not yet exhausted, ignoring this batch.")
            return False, nonfinite_count
    # Clip gradient norm
    torch.nn.utils.clip_grad_norm_(
        (p for p in modules.parameters()), max_grad_norm
    )
    return True, nonfinite_count

def update_average(loss, avg_loss, step):
    if torch.isfinite(loss):
        avg_loss -= avg_loss / step
        avg_loss += float(loss) / step
    return avg_loss

def parse_args():
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
        help="One of {nccl, gloo, mpi}",
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

    return parser.parse_args()
