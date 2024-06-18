# LoRa

A simple PyTorch LoRa implementation for educational purpose, works only with Linear Layers.

# Usage example

    from transformers import AutoModelForCausalLM
    from utils import update_decoder_layer, freeze_model, print_model_parameters
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )

    freeze_model(model)

    update_decoder_layer(model.model.layers)

    print_model_parameters(model)

# To Do

- [ ] Add support for other architectures
- [ ] Add an efficient merge mechanism for adapters



