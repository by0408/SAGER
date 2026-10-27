import torch
from .common import model_identity, parse_layers


def hidden_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Unsupported decoder block output: " + str(type(output)))


def replace_hidden(output, hidden):
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    raise TypeError(type(output))


class Adapter:
    """Raw decoder block outputs; no assumptions about LM-head input dimension."""
    def __init__(self, model):
        self.model = model
        self.family = model.config.model_type
        if self.family == "opt":
            self.blocks = model.model.decoder.layers
        elif self.family in ("llama", "mistral", "gemma2"):
            self.blocks = model.model.layers
        else:
            raise ValueError("Supported: opt, mistral, llama, gemma2; got " + self.family)
        self.dimension = int(model.config.hidden_size)
        self.input_device = model.get_input_embeddings().weight.device

    def layers(self, cfg):
        return parse_layers(cfg["layers"], len(self.blocks))

    def describe(self):
        return {"family": self.family, "dimension": self.dimension, "num_blocks": len(self.blocks),
                "edit_site": "1-based decoder block output, last position, before subsequent blocks/final norm",
                "head_dimension": int(self.model.get_output_embeddings().weight.shape[1])}


def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs pad or EOS token")
    kwargs = dict(local_files_only=True, torch_dtype=dtype, attn_implementation="eager")
    device = cfg.get("device_map", "auto")
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], **kwargs).to("cpu")
    else:
        kwargs["device_map"] = device
        if cfg.get("max_memory"):
            kwargs["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in cfg["max_memory"].items()}
        model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], **kwargs)
    model.eval().requires_grad_(False)
    adapter = Adapter(model)
    if cfg.get("family") and cfg["family"] != adapter.family:
        raise ValueError("Configured family does not match loaded model")
    return model, tokenizer, adapter, model_identity(cfg["model_path"])
