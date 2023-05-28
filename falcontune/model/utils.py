from transformers.utils import logging

logger = logging.get_logger("transformers")


def model_to_half(model, cast_model=True):
    if cast_model:
        model.half()

    for n, m in model.named_modules():
        if m.__class__.__name__ == 'QuantLinear':
            logger.debug(f'Converting to half {n}.')
            m.scales = m.scales.half()
            m.bias = m.bias.half() if (m.bias is not None) else None
    logger.info('Converted as Half.')
