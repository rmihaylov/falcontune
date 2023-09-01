# falcontune: 4-Bit Finetuning of FALCONs on a Consumer GPU

**falcontune** allows finetuning FALCONs (e.g., falcon-40b-4bit) on as little as one consumer-grade A100 40GB. 

Its features tiny and easy-to-use codebase.

One benefit of being able to finetune larger LLMs on one GPU is the ability to easily leverage data parallelism for large models.

Underneath the hood, **falcontune** implements the LoRA algorithm over an LLM compressed using the GPTQ algorithm, which requires implementing a backward pass for the quantized LLM.

**falcontune** can generate a 50-token recipe on A100 40GB for ~ 10 seconds using triton backend

```
$ falcontune generate --interactive --model falcon-40b-instruct-4bit --weights gptq_model-4bit--1g.safetensors --max_new_tokens=50 --use_cache --do_sample --prompt "How to prepare pasta?"


How to prepare pasta?
Here's a simple recipe to prepare pasta:

Ingredients:
- 1 pound of dry pasta
- 4-6 cups of water
- Salt (optional)

Instructions:
1. Boil the water

Took 10.042 s
```

This example is based on the model: TheBloke/falcon-40b-instruct-GPTQ.

Here is a [Google Colab](https://colab.research.google.com/drive/1Pv7Dn60u_ANgkhRojAIX-VOkU3J-2cYh?usp=sharing). 
You will need a A100 40GB to finetune the model.

## Installation

### Setup

```
pip install -r requirements.txt 
python setup.py install         
```

The default backend is triton which is the fastest. For cuda support install also the CUDA kernels:

```
python setup_cuda.py install         
```


## Running falcontune

The above process installs a `falcontune` command in your environment.

### Download Models

First, start by downloading the weights of a FALCON model:
```
$ wget https://huggingface.co/TheBloke/falcon-40b-instruct-GPTQ/resolve/main/model.safetensors
```

### Generate Text

You can generate text directly from the command line. This generates text from the base model:
```
$ falcontune generate \
    --interactive \
    --model falcon-40b-instruct-4bit \
    --weights gptq_model-4bit--1g.safetensors \
    --max_new_tokens=50 \
    --use_cache \
    --do_sample \
    --instruction "Who was the first person on the moon?"
```

### Finetune A Base Model

You may also finetune a base model yourself. First, you need to download a dataset:
```
$ wget https://github.com/gururise/AlpacaDataCleaned/raw/main/alpaca_data_cleaned.json
```

You can finetune any model of the FALCON family:

<details>
<summary>FALCON-7B</summary>
<br>

    $ falcontune finetune \
        --model=falcon-7b \
        --weights=tiiuae/falcon-7b \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-7b-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-7b-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-7b \
        --weights tiiuae/falcon-7b \
        --lora_apply_dir falcon-7b-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>


<details>
<summary>FALCON-7B-INSTRUCT</summary>
<br>

    $ falcontune finetune \
        --model=falcon-7b-instruct \
        --weights=tiiuae/falcon-7b-instruct \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-7b-instruct-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-7b-instruct-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-7b-instruct \
        --weights mosaicml/falcon-7b-instruct \
        --lora_apply_dir falcon-7b-instruct-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>


<details>
<summary>FALCON-40B</summary>
<br>

    $ falcontune finetune \
        --model=falcon-40b \
        --weights=tiiuae/falcon-40b \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-40b-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-40b-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-40b \
        --weights tiiuae/falcon-40b\
        --lora_apply_dir falcon-40b-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>

<details>
<summary>FALCON-40B-INSTRUCT</summary>
<br>

    $ falcontune finetune \
        --model=falcon-40b-instruct \
        --weights=tiiuae/falcon-40b-instruct \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-40b-instruct-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-40b-instruct-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-40b-instruct \
        --weights tiiuae/falcon-40b-instruct\
        --lora_apply_dir falcon-40b-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>

<details>
<summary>FALCON-7B-INSTRUCT-4BIT</summary>
<br>

    $ wget https://huggingface.co/TheBloke/falcon-7b-instruct-GPTQ/resolve/main/gptq_model-4bit-64g.safetensors
    
    $ falcontune finetune \
        --model=falcon-7b-instruct-4bit \
        --weights=gptq_model-4bit-64g.safetensors \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-7b-instruct-4bit-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-7b-instruct-4bit-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-7b-instruct-4bit \
        --weights gptq_model-4bit-64g.safetensors \
        --lora_apply_dir falcon-7b-instruct-4bit-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>

<details>
<summary>FALCON-40B-INSTRUCT-4BIT</summary>
<br>

    $ wget https://huggingface.co/TheBloke/falcon-40b-instruct-GPTQ/resolve/main/gptq_model-4bit--1g.safetensors
    
    $ falcontune finetune \
        --model=falcon-40b-instruct-4bit \
        --weights=gptq_model-4bit--1g.safetensors \
        --dataset=./alpaca_data_cleaned.json \
        --data_type=alpaca \
        --lora_out_dir=./falcon-40b-instruct-4bit-alpaca/ \
        --mbatch_size=1 \
        --batch_size=2 \
        --epochs=3 \
        --lr=3e-4 \
        --cutoff_len=256 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --warmup_steps=5 \
        --save_steps=50 \
        --save_total_limit=3 \
        --logging_steps=5 \
        --target_modules='["query_key_value"]'

    The above commands will download the model and use LoRA to finetune the quantized model. The final adapters and the checkpoints will be saved in `falcon-40b-instruct-4bit-alpaca` and available for generation as follows:

    $ falcontune generate \
        --interactive \
        --model falcon-40b-instruct-4bit \
        --weights gptq_model-4bit--1g.safetensors \
        --lora_apply_dir falcon-40b-instruct-4bit-alpaca \
        --max_new_tokens 50 \
        --use_cache \
        --do_sample \
        --instruction "How to prepare pasta?"

</details>

## Acknowledgements

**falcontune** is based on the following projects:
* The GPTQ algorithm and codebase by the [IST-DASLAB](https://github.com/IST-DASLab/gptq) with modifications by [@qwopqwop200](https://github.com/qwopqwop200/)
* The `alpaca_lora_4bit` repo by [johnsmith0031](https://github.com/johnsmith0031)
* The PEFT repo and its implementation of LoRA
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium
* The `llmtune` repo by [kuleshov-group](https://github.com/kuleshov-group/llmtune)


## Consultations
Need a custom solution? Let me know: `r.m.mihaylov@gmail.com`
