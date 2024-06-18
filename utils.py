from functools import partial

from lora_layer import LoRaLayer


def update_decoder_layer(layers, rank=4, alpha=1.0):
    lora_layer = partial(LoRaLayer, rank=rank, alpha=alpha)
    for idx, _ in enumerate(layers):
        layers[idx].self_attn.o_proj = lora_layer(layers[idx].self_attn.o_proj)
        layers[idx].self_attn.qkv_proj = lora_layer(layers[idx].self_attn.qkv_proj)
        layers[idx].mlp.gate_up_proj = lora_layer(layers[idx].mlp.gate_up_proj)
        layers[idx].mlp.down_proj = lora_layer(layers[idx].mlp.down_proj)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
