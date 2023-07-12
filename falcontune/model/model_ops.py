import importlib

import torch
from peft import PeftModel
import accelerate
import transformers
from transformers.models.falcon.modeling_falcon import FalconForCausalLM, FalconConfig, FalconDecoderLayer
from transformers.utils import logging

from falcontune.backend.base import replace_4bit_linear, find_layers
from falcontune.model.lora import Linear4bitLt


logger = logging.get_logger("transformers")


def load_model(llm_config, checkpoint, half=False, backend='triton'):
    config = FalconConfig.from_pretrained(llm_config.hf_config_name)
    config.max_seq_len = llm_config.max_seq_len

    assert config.alibi is False
    assert config.bias is False

    if half:
        torch.set_default_dtype(torch.half)

    if (llm_config.bits == 4) and (llm_config.groupsize is not None):
        with accelerate.init_empty_weights():
            ql = importlib.import_module(f'falcontune.backend.{backend}.quantlinear')

            model = FalconForCausalLM(config)
            model = model.eval()

            layers = find_layers(model)
            del layers['lm_head']

            replace_4bit_linear(
                model,
                layers,
                llm_config.bits,
                llm_config.groupsize,
                quantlinear_class=ql.QuantLinear
            )

        model = accelerate.load_checkpoint_and_dispatch(
            model=model,
            checkpoint=checkpoint,
            device_map=llm_config.device_map,
            no_split_module_classes=["DecoderLayer"]
        )

        model.loaded_in_4bit = True

    elif llm_config.bits == 8:
        model = FalconForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            load_in_8bit=True,
            device_map=llm_config.device_map
        )
        model.loaded_in_8bit = True

    else:
        model = FalconForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=llm_config.device_map
        )
        model.loaded_in_bf16 = True

    model.seqlen = llm_config.max_seq_len

    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_config.hf_tokenizer_config)
    tokenizer.truncation_side = 'left'
    tokenizer.padding_side = 'left'

    tokenizer.bos_token_id = None
    tokenizer.eos_token_id = tokenizer.vocab["<|endoftext|>"]

    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None

    return model, tokenizer


def load_model_and_offload(llm_config, checkpoint, half=False, backend='triton', lora_path=None, max_memory=None):
    if max_memory is None:
        max_memory = {0: '13Gib', 'cpu': '25Gib'}

    config = FalconConfig.from_pretrained(llm_config.hf_config_name)
    config.max_seq_len = llm_config.max_seq_len

    assert config.alibi is False

    if half:
        torch.set_default_dtype(torch.half)

    with accelerate.init_empty_weights():
        ql = importlib.import_module(f'falcontune.backend.{backend}.quantlinear')

        model = FalconForCausalLM(config)
        model = model.eval()

        layers = find_layers(model)

        for name in ['lm_head']:
            if name in layers:
                del layers[name]

        replace_4bit_linear(
            model,
            layers,
            llm_config.bits,
            llm_config.groupsize,
            quantlinear_class=ql.QuantLinear
        )

    accelerate.load_checkpoint_in_model(
        model,
        checkpoint=checkpoint,
        device_map={'': 'cpu'})

    model.loaded_in_4bit = True

    if lora_path is not None:
        model = PeftModel.from_pretrained(
            model, lora_path,
            device_map={'': 'cpu'},
            torch_dtype=torch.float32,
            is_trainable=True)

        logger.info('{} Lora Applied.'.format(lora_path))

    model.seqlen = llm_config.max_seq_len

    for n, m in model.named_modules():
        if isinstance(m, ql.QuantLinear) or isinstance(m, Linear4bitLt):
            m.scales = m.scales.half()
            m.bias = m.bias.half()

    device_map = accelerate.infer_auto_device_map(
        model, max_memory=max_memory,
        no_split_module_classes=["DecoderLayer"])

    model = accelerate.dispatch_model(
        model, device_map=device_map,
        offload_buffers=True, main_device=0)

    torch.cuda.empty_cache()

    logger.info('Total {:.2f} Gib VRAM used.'.format(torch.cuda.memory_allocated() / 1024 / 1024))

    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_config.hf_config_name)
    tokenizer.truncation_side = 'left'
    tokenizer.padding_side = 'left'

    tokenizer.bos_token_id = None
    tokenizer.eos_token_id = tokenizer.vocab["<|endoftext|>"]

    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None

    return model, tokenizer
