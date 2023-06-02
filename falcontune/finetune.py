import os
import torch

import wandb
import transformers
from transformers.utils import logging
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from falcontune.data import load_data
from falcontune.model import load_model
from falcontune.model.lora import load_adapter
from falcontune.model.utils import model_to_half

logger = logging.get_logger("transformers")


class FinetuneConfig:
    def __init__(self, args):
        self.__dict__.update(args.__dict__)

        self.target_modules = eval(self.target_modules)
        self.gradient_accumulation_steps = self.batch_size // self.mbatch_size
        self.lora_dropout = 0 if self.gradient_checkpointing else self.lora_dropout  # should be 0 if gradient checkpointing is on
        self.val_set_size = int(self.val_set_size) if self.val_set_size > 1.0 else float(self.val_set_size)

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.local_rank))
        self.ddp = self.world_size != 1
        self.device_map = "auto" if not self.ddp else {"": self.local_rank}

        if self.ddp:
            self.gradient_accumulation_steps = self.gradient_accumulation_steps // self.world_size

    def __str__(self) -> str:
        s = f"\nParameters:\n{'config':-^20}\n{self.dataset=}\n{self.data_type=}\n{self.lora_out_dir=}\n{self.lora_apply_dir=}" + \
            f"\n{self.weights=}\n{self.target_modules=}\n\n" + \
            f"{'training':-^20}\n" + \
            f"{self.mbatch_size=}\n{self.batch_size=}\n{self.gradient_accumulation_steps=}\n{self.epochs=}\n{self.lr=}\n{self.cutoff_len=}\n" + \
            f"{self.lora_r=}\n{self.lora_alpha=}\n{self.lora_dropout=}\n{self.val_set_size=}\n" + \
            f"{self.gradient_checkpointing=}\n{self.gradient_checkpointing_ratio=}\n" + \
            f"{self.warmup_steps=}\n{self.save_steps=}\n{self.save_total_limit=}\n" + \
            f"{self.logging_steps=}\n" + \
            f"{self.checkpoint=}\n{self.skip=}\n" + \
            f"{self.world_size=}\n{self.ddp=}\n{self.device_map=}\n"
        return s.replace("self.", "")


def finetune(args):
    llm, tokenizer = load_model(args.model, args.weights, backend=args.backend)
    tune_config = FinetuneConfig(args)

    transformers.logging.set_verbosity_info()

    # * Show loaded parameters
    if tune_config.local_rank == 0:
        logger.info(f"{tune_config}\n")

    if tune_config.gradient_checkpointing:
        logger.info('Disable Dropout.')

    if tune_config.mbatch_size > tune_config.batch_size:
        raise Exception('batch_size need to be larger than mbatch_size.')

    lora_config = LoraConfig(
        r=tune_config.lora_r,
        lora_alpha=tune_config.lora_alpha,
        target_modules=tune_config.target_modules,
        lora_dropout=tune_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = load_adapter(
        llm,
        lora_apply_dir=tune_config.lora_apply_dir,
        lora_config=lora_config,
        ddp=tune_config.ddp
    )

    if getattr(model, 'loaded_in_4bit', False):
        model_to_half(model, cast_model=False)

    model.print_trainable_parameters()

    if not tune_config.skip:
        # Load Data
        data = load_data(tune_config, tokenizer)

        # Use gradient checkpointing
        if tune_config.gradient_checkpointing:
            logger.info('Applying gradient checkpointing ...')
            from falcontune.model.gradient_checkpointing import apply_gradient_checkpointing
            from falcontune.model.falcon.model import get_decoder_layer

            apply_gradient_checkpointing(
                model,
                decoder_layer_class=get_decoder_layer(num_heads=llm.config.n_head),
                checkpoint_ratio=tune_config.gradient_checkpointing_ratio)

        # Disable Trainer's DataParallel for multigpu
        if not tune_config.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        # Count eval count for wandb
        if tune_config.val_set_size > 0:
            eval_count = 10
            eval_steps = max(
                tune_config.logging_steps,
                (len(data.train_data) + len(data.val_data)) // (eval_count * tune_config.mbatch_size)
            )
            logger.info(f"Run eval every {eval_steps} steps")
        else:
            eval_steps = 0

        training_arguments = transformers.TrainingArguments(
            per_device_train_batch_size=tune_config.mbatch_size,
            gradient_accumulation_steps=tune_config.gradient_accumulation_steps,
            warmup_steps=tune_config.warmup_steps,
            optim="adamw_torch",
            num_train_epochs=tune_config.epochs,
            learning_rate=tune_config.lr,
            fp16=True,
            logging_steps=tune_config.logging_steps,
            evaluation_strategy="steps" if eval_steps != 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if eval_steps != 0 else None,
            save_steps=tune_config.save_steps,
            output_dir=tune_config.lora_out_dir,
            save_total_limit=tune_config.save_total_limit,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if tune_config.ddp else None,
        )

        trainer = transformers.Trainer(
            model=model,
            train_dataset=data.train_data,
            eval_dataset=data.val_data,
            args=training_arguments,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        model.config.use_cache = False

        # Set Model dict
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

        # Set Verbose
        if tune_config.verbose:
            transformers.logging.set_verbosity_info()

        # Run Trainer
        with wandb.init(project="alpaca_lora_4bit") as run:
            if tune_config.resume_checkpoint:
                logger.info('Resuming from {} ...'.format(tune_config.resume_checkpoint))
                state_dict_peft = torch.load(os.path.join(tune_config.resume_checkpoint, 'pytorch_model.bin'), map_location='cpu')
                set_peft_model_state_dict(model, state_dict_peft)
                trainer.train(tune_config.resume_checkpoint)
            else:
                trainer.train()

        # Restore old model state dict
        model.state_dict = old_state_dict

        logger.info('Train completed.')

    # Save Model
    model.save_pretrained(tune_config.lora_out_dir)

    if tune_config.checkpoint:
        logger.info("Warning: Merge model + LoRA and save the whole checkpoint not implemented yet.")

    logger.info('Model Saved.')
