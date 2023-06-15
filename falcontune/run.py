import os
import argparse

from falcontune.finetune import finetune
from falcontune.generate import generate

from falcontune.model import MODEL_CONFIGS
from falcontune.backend import BACKENDS
from falcontune.data import DATA_TYPES


def get_args():
    parser = argparse.ArgumentParser(
        prog=__file__.split(os.path.sep)[-1],
        description="Produce FALCON in 4-bit training"
    )

    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers(title='Commands')

    # GENERATE
    gen_parser = subparsers.add_parser('generate')
    gen_parser.set_defaults(func=generate)
    gen_parser.add_argument('--model', choices=MODEL_CONFIGS, required=True, help='Type of model to load')
    gen_parser.add_argument('--weights', type=str, required=True, help='Path to the base model weights.')
    gen_parser.add_argument("--lora_apply_dir", default=None, required=False, help="Path to directory from which LoRA has to be applied before training. Default: %(default)s")
    gen_parser.add_argument('--prompt', type=str, default='', help='Text used to initialize generation')
    gen_parser.add_argument('--instruction', type=str, default='', help='Instruction for an alpaca-style model')
    gen_parser.add_argument('--max_new_tokens', type=int, default=400, help='Maximum new tokens of the sequence to be generated.')
    gen_parser.add_argument('--top_p', type=float, default=.95, help='Top p sampling parameter.')
    gen_parser.add_argument('--top_k', type=int, default=40, help='Top p sampling parameter.')
    gen_parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature.')
    gen_parser.add_argument('--use_cache', action="store_true", help='Use cache when generating.')
    gen_parser.add_argument('--do_sample', action="store_true", help='Sampling when generating.')
    gen_parser.add_argument('--num_beams', type=int, default=1, help='Number of beams.')
    gen_parser.add_argument('--interactive', action="store_true", help='Enter prompts interactively.')
    gen_parser.add_argument('--backend', type=str, default='triton', choices=BACKENDS, required=False, help='Change the default backend.')
    gen_parser.add_argument('--contextual', action="store_true", help='Use contextual generation.')
    gen_parser.add_argument('--input', type=str, default='', help='Input for contextual generation.')

    # FINETUNE
    tune_parser = subparsers.add_parser('finetune')
    tune_parser.set_defaults(func=finetune)

    # Model args group
    tune_parser.add_argument('--model', choices=MODEL_CONFIGS, required=True, help='Type of model to load')
    tune_parser.add_argument('--weights', type=str, required=True, help="Path to the quantized model in huggingface format. Default: %(default)s")
    tune_parser.add_argument("--data_type", choices=DATA_TYPES, help="Dataset format", default="alpaca")
    tune_parser.add_argument("--dataset", required=False, help="Path to local dataset file.")
    tune_parser.add_argument("--lora_out_dir", default="alpaca_lora", required=False, help="Directory to place new LoRA. Default: %(default)s")
    tune_parser.add_argument("--lora_apply_dir", default=None, required=False, help="Path to directory from which LoRA has to be applied before training. Default: %(default)s")
    tune_parser.add_argument("--resume_checkpoint", default=None, type=str, required=False, help="Path to checkpoint to resume training from. Default: %(default)s")

    # Training args group
    tune_parser.add_argument("--mbatch_size", default=1, type=int, help="Micro-batch size. Default: %(default)s")
    tune_parser.add_argument("--batch_size", default=2, type=int, help="Batch size. Default: %(default)s")
    tune_parser.add_argument("--epochs", default=3, type=int, help="Epochs. Default: %(default)s")
    tune_parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate. Default: %(default)s")
    tune_parser.add_argument("--cutoff_len", default=256, type=int, help="Default: %(default)s")
    tune_parser.add_argument("--lora_r", default=8, type=int, help="Default: %(default)s")
    tune_parser.add_argument("--lora_alpha", default=16, type=int, help="Default: %(default)s")
    tune_parser.add_argument("--lora_dropout", default=0.05, type=float, help="Default: %(default)s")
    tune_parser.add_argument("--gradient_checkpointing", action="store_true", required=False, help="Use gradient checkpoint. Default: %(default)s")
    tune_parser.add_argument("--gradient_checkpointing_ratio", default=1, type=float, help="Gradient checkpoint ratio. Default: %(default)s")
    tune_parser.add_argument("--val_set_size", default=0.2, type=float, help="Validation set size. Default: %(default)s")
    tune_parser.add_argument("--warmup_steps", default=50, type=int, help="Default: %(default)s")
    tune_parser.add_argument("--save_steps", default=50, type=int, help="Default: %(default)s")
    tune_parser.add_argument("--save_total_limit", default=3, type=int, help="Default: %(default)s")
    tune_parser.add_argument("--logging_steps", default=10, type=int, help="Default: %(default)s")
    tune_parser.add_argument("-c", "--checkpoint", action="store_true", help="Produce checkpoint instead of LoRA. Default: %(default)s")
    tune_parser.add_argument("--skip", action="store_true", help="Don't train model. Can be useful to produce checkpoint from existing LoRA. Default: %(default)s")
    tune_parser.add_argument("--verbose", action="store_true", help="If output log of training. Default: %(default)s")
    tune_parser.add_argument("--target_modules", default="['q_proj', 'v_proj']", type=str, help="Target modules for LoRA.")

    # Backend
    tune_parser.add_argument('--backend', type=str, default='triton', choices=BACKENDS, required=False, help='Change the default backend.')

    # Data args
    tune_parser.add_argument("--use_eos_token", default=1, type=int, help="Use eos token instead if padding with 0. enable with 1, disable with 0.")

    # Multi GPU Support
    tune_parser.add_argument("--local_rank", type=int, default=0, help="local rank if using torch.distributed.launch")

    return parser.parse_args()


def main():
    args = get_args()
    args.func(args)


if __name__ == '__main__':
    main()
