"""Greedy, cache-backed decoding for the independently versioned GPU editor.

Tokenization is intentionally the same individual native-tokenizer call as the
original challenge runner.  Padding is constructed here without changing the
tokenizer, because the original PPL scorer also uses that tokenizer.  A batch
can change FP16 model arithmetic; this module does not claim bitwise agreement
between batch sizes.  EOS rows remain physical batch rows but never receive
editing hooks or returned tokens after their first EOS.
"""

from contextlib import nullcontext
import hashlib
import inspect
import json
import time

import torch


def _digest_ids(ids):
    # Identical JSON representation to sager.common.digest(list[int]).
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


def _position_ids(attention_mask):
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 0)


def _eos_ids(tokenizer):
    value = tokenizer.eos_token_id
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [int(item) for item in value]
    return [int(value)]


def _input_device(model, adapter, requested):
    if requested is not None:
        return torch.device(requested)
    if adapter is not None:
        return adapter.input_device
    return model.get_input_embeddings().weight.device


def _prepare_inputs(model, full_ids, attention_mask, position_ids, past,
                    cache_position, forward_parameters, forward_variadic):
    """Let HF handle its cache type and slicing, never inspect cache internals."""
    prepare = getattr(model, "prepare_inputs_for_generation", None)
    kwargs = dict(input_ids=full_ids, attention_mask=attention_mask,
                  position_ids=position_ids, past_key_values=past,
                  cache_position=cache_position, use_cache=True)
    if callable(prepare):
        result = dict(prepare(**kwargs))
    else:
        # Useful for simple compatible causal modules in the verification tests.
        result = kwargs
        if past is not None:
            result["input_ids"] = full_ids[:, -1:]
            result["position_ids"] = position_ids[:, -1:]
    result["return_dict"] = True
    result["use_cache"] = True
    if not forward_variadic:
        # OPT derives positions from the mask and has no explicit position_ids
        # argument.  LLaMA/Mistral receive the explicit cumulative positions.
        result = {name: value for name, value in result.items()
                  if name in forward_parameters}
    return result


def _validate_records(records, layers, token_counts):
    if len(records) != len(token_counts):
        raise ArithmeticError("Editor exported a different batch size")
    expected_layers = sorted(int(layer) for layer in layers)
    for sample, (rows, count) in enumerate(zip(records, token_counts)):
        if len(rows) != count * len(expected_layers):
            raise ArithmeticError("Missing per-token/layer check for row %d" % sample)
        for step in range(count):
            selected = sorted(int(row["layer"]) for row in rows
                              if int(row["step0"]) == step)
            if selected != expected_layers:
                raise ArithmeticError("Wrong checked layers for row %d, step %d" %
                                      (sample, step))


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, editor=None, adapter=None,
                   input_device=None, max_new_tokens=10, progress_callback=None,
                   record_rotations=False):
    """Generate raw greedy continuations and return rows plus honest batch timing.

    ``prompts`` is a nonempty sequence of raw strings.  ``editor``, if supplied,
    implements ``active()``, ``set_alive([B] bool)`` and ``export_records()``.
    Before every model forward, it receives the rows still predicting tokens.
    No sampling, repetition penalty, chat template, truncation or thread setting
    is introduced here.  The caller controls the single-thread worker policy.

    Callback (optional, once per predicted token)::

        callback({"step": n, "batch_size": B, "rows": [
            {"continuation": text, "new_token_ids": ids,
             "token_count": len(ids), "alive": bool}, ...]})

    Each row's ``generation_seconds`` is shared batch wall time, explicitly
    labelled.  ``timing.per_row_amortized_seconds`` is throughput accounting,
    not a measurement of standalone prompt latency.  Editor records are moved
    to CPU only after the editing context closes.  Callers may safely score PPL
    with the same unedited model after this function returns.
    """
    prompts = list(prompts)
    if not prompts or not all(isinstance(prompt, str) and prompt.strip()
                              for prompt in prompts):
        raise ValueError("Expected nonempty raw prompt strings")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if getattr(model, "training", False):
        raise ValueError("Greedy experiment requires model.eval()")
    if editor is None and adapter is not None:
        if any(block._forward_hooks for block in adapter.blocks):
            raise RuntimeError("Base generation has editing hooks")
    if record_rotations and (editor is None or not getattr(editor, "record_rotations", False)):
        raise ValueError("Full rotation traces require an editor configured to record them")

    device = _input_device(model, adapter, input_device)
    eos = _eos_ids(tokenizer)
    pad = tokenizer.pad_token_id
    if pad is None:
        if not eos:
            raise ValueError("Tokenizer needs a pad or EOS token")
        pad = eos[0]
    pad = int(pad)
    encoded = []
    for prompt in prompts:
        # Do not set add_special_tokens=False: retain the tokenizer's native BOS.
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Tokenizer did not return one nonempty prompt")
        encoded.append(ids[0].long().cpu())
    counts = [int(ids.numel()) for ids in encoded]
    width, batch_size = max(counts), len(encoded)
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit is not None and width + max_new_tokens > int(limit):
        raise ValueError("Context exceeded; refuse truncation")
    full_ids = torch.full((batch_size, width), pad, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(full_ids)
    for sample, ids in enumerate(encoded):
        full_ids[sample, -ids.numel():] = ids.to(device=device)
        attention_mask[sample, -ids.numel():] = 1
    parameters = inspect.signature(model.forward).parameters
    forward_variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in parameters.values())
    forward_parameters = set(parameters)
    alive = torch.ones(batch_size, dtype=torch.bool, device=device)
    history = torch.full((batch_size, max_new_tokens), pad,
                         dtype=torch.long, device=device)
    token_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    eos_tensor = torch.tensor(eos, dtype=torch.long, device=device)
    past, predicted_steps = None, 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started, cpu_started = time.perf_counter(), time.process_time()
    context = editor.active() if editor is not None else nullcontext()
    with context:
        for step in range(max_new_tokens):
            if editor is not None:
                editor.set_alive(alive)
            # Physical cache positions include left padding.  Logical token
            # positions come from each row's mask; no tuple-cache assumptions.
            cache_position = (torch.arange(width, device=device, dtype=torch.long)
                              if past is None else
                              torch.tensor([width + step - 1], device=device,
                                           dtype=torch.long))
            kwargs = _prepare_inputs(model, full_ids, attention_mask,
                                     _position_ids(attention_mask), past,
                                     cache_position, forward_parameters,
                                     forward_variadic)
            output = model(**kwargs)
            logits = output.logits
            if logits.ndim != 3 or logits.shape[0] != batch_size:
                raise ValueError("Expected causal logits [batch, sequence, vocab]")
            # Matches HF greedy argmax (casting logits does not alter argmax).
            selected = logits[:, -1, :].float().argmax(dim=-1)
            selected = torch.where(alive, selected, torch.full_like(selected, pad))
            history[:, step] = selected
            token_counts = token_counts + alive.long()
            if eos:
                is_eos = (selected[:, None] == eos_tensor[None, :]).any(dim=-1)
                alive = alive & ~is_eos
            predicted_steps = step + 1
            if progress_callback is not None:
                cpu_counts = token_counts.cpu().tolist()
                cpu_history = history[:, :predicted_steps].cpu().tolist()
                cpu_alive = alive.cpu().tolist()
                progress_callback({"step": predicted_steps, "batch_size": batch_size,
                                   "rows": [{"continuation": tokenizer.decode(
                                       row[:count], skip_special_tokens=True),
                                       "new_token_ids": row[:count],
                                       "token_count": count, "alive": bool(live)}
                                       for row, count, live in zip(cpu_history,
                                                                  cpu_counts,
                                                                  cpu_alive)]})
            if predicted_steps == max_new_tokens or not bool(alive.any().item()):
                break
            past = output.past_key_values
            if past is None:
                raise RuntimeError("Model did not return a KV cache")
            # Finished rows remain fixed-size dummy rows, have zero mask at new
            # positions, and are inactive in every subsequent editing hook.
            next_ids = torch.where(alive, selected, torch.full_like(selected, pad))
            full_ids = torch.cat((full_ids, next_ids[:, None]), dim=-1)
            attention_mask = torch.cat((attention_mask, alive.long()[:, None]), dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - started
    token_counts_cpu = token_counts.cpu().tolist()
    history_cpu = history[:, :predicted_steps].cpu().tolist()
    export_started = time.perf_counter()
    records = editor.export_records() if editor is not None else [[] for _ in prompts]
    rotations = (editor.export_rotations() if record_rotations else None)
    backend_status = editor.backend_status() if editor is not None else None
    if editor is not None:
        _validate_records(records, editor.layers, token_counts_cpu)
        if any(value != predicted_steps for value in editor.calls.values()):
            raise ArithmeticError("Editor layer call count differs from decoder forwards")
    export_seconds = time.perf_counter() - export_started
    seconds = time.perf_counter() - started
    rows = []
    for sample, (ids, count) in enumerate(zip(history_cpu, token_counts_cpu)):
        tokens = ids[:count]
        row = {"continuation": tokenizer.decode(tokens, skip_special_tokens=True),
               "new_token_ids": tokens, "new_token_ids_sha256": _digest_ids(tokens),
               "token_count": count, "hit_max_new_tokens": count == max_new_tokens,
               "ended_with_eos": bool(tokens and tokens[-1] in eos),
               "prompt_token_count": counts[sample],
               "prompt_token_ids_sha256": _digest_ids(encoded[sample].tolist()),
               "layers": records[sample], "generation_seconds": seconds,
               "generation_timing_scope": "shared_batch_wall_time",
               "amortized_generation_seconds": seconds / batch_size}
        if rotations is not None:
            row["rotations_trace"] = rotations[sample]
        rows.append(row)
    return {"rows": rows, "timing": {
        "batch_wall_seconds": seconds, "decode_wall_seconds": decode_seconds,
        "record_export_seconds": export_seconds, "batch_size": batch_size,
        "predicted_steps": predicted_steps, "prompt_token_counts": counts,
        "padded_prompt_width": width,
        "cpu_process_seconds": time.process_time() - cpu_started,
        "per_row_amortized_seconds": seconds / batch_size,
        "position_ids": "cumsum(attention_mask)-1, masked positions zero",
        "finished_row_policy": "inactive hooks, dummy pad input, zero new-position mask",
        "cache": "model-native past_key_values; physical cache_position",
        "tokenization": "individual raw tokenizer call with native special tokens",
        "repetition_penalty": 1.0,
    }, "backend_status": backend_status}
