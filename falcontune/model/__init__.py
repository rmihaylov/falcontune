from falcontune.model.falcon.config import (
    FALCON7B8bitConfig,
    FALCON7BRW8bitConfig,
    FALCON7BInstruct8bitConfig,
    FALCON40B8bitConfig,
    FALCON40BInstruct8bitConfig,
    FALCON1BRW8bitConfig,
    FALCON7B4bitConfig,
    FALCON40B4bitConfig,
)


MODEL_CONFIGS = {
    FALCON7B8bitConfig.name: FALCON7B8bitConfig,
    FALCON7BRW8bitConfig.name: FALCON7BRW8bitConfig,
    FALCON7BInstruct8bitConfig.name: FALCON7BInstruct8bitConfig,
    FALCON40B8bitConfig.name: FALCON40B8bitConfig,
    FALCON40BInstruct8bitConfig.name: FALCON40BInstruct8bitConfig,
    FALCON1BRW8bitConfig.name: FALCON1BRW8bitConfig,
    FALCON7B4bitConfig.name: FALCON7B4bitConfig,
    FALCON40B4bitConfig.name: FALCON40B4bitConfig,
}


def load_model(model_name: str, weights, half=False, backend='triton'):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name: {model_name}")

    model_config = MODEL_CONFIGS[model_name]

    if model_name in MODEL_CONFIGS:
        from falcontune.model.falcon.model import load_model
        model, tokenizer = load_model(model_config, weights, half=half, backend=backend)

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.eval()
    return model, tokenizer
