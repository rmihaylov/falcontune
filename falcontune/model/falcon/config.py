class FALCON7B4bitConfig:
    name = 'falcon-7b-instruct-4bit'
    hf_config_name = "TheBloke/falcon-7b-instruct-GPTQ"
    hf_tokenizer_config = "TheBloke/falcon-7b-instruct-GPTQ"
    bits = 4
    groupsize = 64
    max_seq_len = 2048
    device_map = "auto"


class FALCON40B4bitConfig:
    name = 'falcon-40b-instruct-4bit'
    hf_config_name = "TheBloke/falcon-40b-instruct-GPTQ"
    hf_tokenizer_config = "TheBloke/falcon-40b-instruct-GPTQ"
    bits = 4
    groupsize = -1
    max_seq_len = 2048
    device_map = "auto"


class FALCON7B8bitConfig:
    name = 'falcon-7b'
    hf_config_name = "tiiuae/falcon-7b"
    hf_tokenizer_config = "tiiuae/falcon-7b"
    bits = 8
    groupsize = None
    max_seq_len = 2048
    device_map = "auto"


class FALCON7BInstruct8bitConfig:
    name = 'falcon-7b-instruct'
    hf_config_name = "tiiuae/falcon-7b-instruct"
    hf_tokenizer_config = "tiiuae/falcon-7b-instruct"
    bits = 8
    groupsize = None
    max_seq_len = 2048
    device_map = "auto"


class FALCON7BRW8bitConfig:
    name = 'falcon-rw-7b'
    hf_config_name = "tiiuae/falcon-rw-7b"
    hf_tokenizer_config = "tiiuae/falcon-rw-7b"
    bits = 8
    groupsize = None
    max_seq_len = 2048
    device_map = "auto"


class FALCON1BRW8bitConfig:
    name = 'falcon-rw-1b'
    hf_config_name = "tiiuae/falcon-rw-1b"
    hf_tokenizer_config = "tiiuae/falcon-rw-1b"
    bits = 8
    groupsize = None
    max_seq_len = 2048
    device_map = "auto"


class FALCON40B8bitConfig:
    name = 'falcon-40b'
    hf_config_name = "tiiuae/falcon-40b"
    hf_tokenizer_config = "tiiuae/falcon-40b"
    bits = 8
    groupsize = None
    max_seq_len = 2048
    device_map = "auto"


class FALCON40BInstruct8bitConfig:
    name = 'falcon-40b-instruct'
    hf_config_name = "tiiuae/falcon-40b-instruct"
    hf_tokenizer_config = "tiiuae/falcon-7b-instruct"
    bits = 8
    groupsize = None
    max_seq_len = 2048
    device_map = "auto"
