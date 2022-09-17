import argparse
from al.methods import ALL_METHODS


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    training_args = parser.add_argument_group('training')

    training_args.add_argument('--learning_rate',      type=float, default=1e-3)
    training_args.add_argument('--weight_decay',       type=float, default=1e-2)
    training_args.add_argument('--momentum',           type=float, default=0.9)
    training_args.add_argument('--batch_size',         type=int,   default=64)
    training_args.add_argument('--num_epochs',         type=int,   default=200)
    training_args.add_argument('--optimizer_type',     type=str,   default="sgd", 
        choices=["sgd", "adam", "adamw"])
    training_args.add_argument('--lr_scheduler_type',  type=str,   default="none", 
        choices=["none", "onecycle", "exponential", "cosine"])
    training_args.add_argument('--lr_scheduler_param', type=float, default=10.0)

    training_args.add_argument('--log_every',  type=float, default=10)
    training_args.add_argument('--eval_every', type=float, default=10)

    return parser


def add_swa_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    swa_args = parser.add_argument_group('swa')

    swa_args.add_argument('--swa_start',          type=int,   default=100)
    swa_args.add_argument('--swa_anneal_epochs',  type=int,   default=50)
    swa_args.add_argument('--swa_lr_multiplier',  type=float, default=10.0)
    swa_args.add_argument('--swa_scheduler_type', type=str,   default="constant",
        choices=["none", "constant", "cosine"])
    swa_args.add_argument('--start_swa_at_end', action='store_true')

    return parser


def add_query_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    query_args = parser.add_argument_group('query')
    
    query_args.add_argument('--num_episodes',    type=int, default=30)
    query_args.add_argument('--num_ensembles',   type=int, default=5)
    query_args.add_argument('--query_size',      type=int, default=500)
    query_args.add_argument('--query_type',      type=str, default="random", choices=ALL_METHODS)
    query_args.add_argument('--init_query_size', type=int, default=500)
    query_args.add_argument('--init_query_type', type=str, default="random", choices=ALL_METHODS)
    query_args.add_argument('--eval_query_size', type=int, default=500)
    query_args.add_argument('--eval_query_type', type=str, default="random", choices=ALL_METHODS)

    return parser