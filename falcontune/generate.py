import time
import torch

from transformers.utils import logging

from falcontune.data import make_prompt
from falcontune.model import load_model
from falcontune.model.lora import load_adapter
from falcontune.model.utils import model_to_half

logger = logging.get_logger("transformers")


class AMPWrapper:
    def __init__(self, model, options=None):
        self.model = model
        self.options = options
        if self.options is None:
            self.options = {'enabled': True, 'device_type': 'cuda'}

    def autocast_forward(self, *args, **kwargs):
        with torch.amp.autocast(**self.options):
            return self.model.non_autocast_forward(*args, **kwargs)

    def autocast_generate(self, *args, **kwargs):
        with torch.amp.autocast(**self.options):
            return self.model.non_autocast_generate(*args, **kwargs)

    def apply_forward(self):
        self.model.non_autocast_forward = self.model.forward
        self.model.forward = self.autocast_forward

    def apply_generate(self):
        self.model.non_autocast_generate = self.model.generate
        self.model.generate = self.autocast_generate


def format_output(raw_output):
    return raw_output.split("### Response:")[1].strip()


def generate(args):
    model, tokenizer = load_model(
        args.model,
        args.weights,
        backend=args.backend)

    if args.lora_apply_dir is not None:
        model = load_adapter(model, lora_apply_dir=args.lora_apply_dir)

    if getattr(model, 'loaded_in_4bit', False):
        model_to_half(model)

    logger.debug('Apply AMP Wrapper ...')
    wrapper = AMPWrapper(model)
    wrapper.apply_generate()

    if args.prompt and args.instruction:
        raise Exception('Cannot specify both prompt and instruction')

    prompt, instruction, input_ = args.prompt, args.instruction, args.input
    running_input = input_ if input_ else "" # used as context in interactive mode
    is_contextual = args.contextual if args.contextual else False

    while True:
        prompt = make_prompt(instruction, input_= running_input) \
            if args.instruction else prompt

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_ids,
                do_sample=args.do_sample,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                use_cache=args.use_cache,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=args.num_beams
            )
        end_time = time.time()

        output = tokenizer.decode(generated_ids.cpu().tolist()[0], skip_special_tokens=True)

        if args.instruction:
            output = format_output(output)

        print('\n\n\n')
        print(output)
        print(f'\nTook {round(end_time - start_time, 3)} s\n\n\n\n')

        if not args.interactive:
            break

        if is_contextual:
            running_input += '\n{output}'.format(output=output)
            # prompt for new input
            new_input = input("Enter new input: ")
            running_input += '\n{new_input}'.format(new_input=new_input)
        else:
            if args.instruction:
                instruction = input("Enter new instruction: ")
            else:
                prompt = input("Enter new prompt: ")
