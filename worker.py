import argparse
import asyncio
from contextlib import asynccontextmanager
import datetime
import json
import uuid
from fastapi import *
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
import psycopg
from transformer_lens import HookedTransformer, utils
import torch as t
import circuitsvis as cv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import einops
import functools
from plotly.express import imshow
import numpy as np
import requests as rqsts
import plotly.io as pio
import aio_pika
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, required=True)
args = parser.parse_args()

GPU_ID = args.gpu_id

def to_numpy(
    tensor: t.Tensor
):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")
def convert_tokens_to_string(model, tokens, batch_index=0):
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

def ablation(
    data: dict
):
    def get_log_probs(logits, tokens):  
        logprobs = logits.log_softmax(dim=-1)
        
        target_tokens = tokens[:, 1:]            
        logprobs = logprobs[:, :-1, :]
        
        correct_logprobs = t.gather(
            logprobs, 
            dim=-1, 
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        return correct_logprobs
        
    def head_zero_ablation_hook(z, hook, head_index_to_ablate):
        z[:, :, head_index_to_ablate, :] = 0.0
        
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
    
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    prompt = config["prompt"]
    
    ablation_scores = t.zeros(len(config["head_n"]), device=device)
    tokens = model.to_tokens(prompt)
    
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()
    
    for i, head_idx in enumerate(config["head_n"]):
        temp_hook_fn = functools.partial(head_zero_ablation_hook, head_index_to_ablate=head_idx)
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", config["layer_n"]), temp_hook_fn)])
        loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1) :].mean()
    
        ablation_scores[i] = loss - loss_no_ablation
    
    ablation_scores = ablation_scores.unsqueeze(0).transpose(-1,-2).cpu().detach().numpy()
    
    fig = imshow(
        ablation_scores,
        y=config["head_n"],
        labels={"y": "Голова внимания", "color": "Logit diff"},
        title="Изменения значения функции потерь после аблации",
        text_auto=".2f",
    )

    content = {
        "scores": ablation_scores.squeeze(1).tolist(),
        "fig": fig.to_json()
    }

    return content

def activation_map(
    data: dict  
):
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
    
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    prompt = config["prompt"]
    
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    attention_pattern = cache["pattern", config["layer_n"]]
    str_tokens = model.to_str_tokens(prompt)
    
    fig = cv.attention.attention_patterns(
        str_tokens,
        attention_pattern[tuple([h for h in config['head_n']]),:,:]
    )
    
    current_token = []
    prev_token = []
    first_token = []
    induction = []
    for head in config["head_n"]:
        attention_pattern = cache["pattern", config["layer_n"]][head]
        score_current = attention_pattern.diagonal().mean()
        score_prev = attention_pattern.diagonal(-1).mean()
        score_first = attention_pattern[:, 0].mean()
        
        seq_len = (attention_pattern.shape[-1] - 1) // 2
        score_induction = attention_pattern.diagonal(-seq_len + 1).mean()
        if score_current > 0.4:
            current_token.append(head)
        if score_prev > 0.4:
            prev_token.append(head)
        if score_first > 0.4:
            first_token.append(head)
        if score_induction > 0.4:
            induction.append(head)

    content = {
        "fig": str(fig._repr_html_()),
        "current": current_token,
        "prev": prev_token,
        "first": first_token,
        "induction": induction
    }

    return content

def logit_attribution(
    data: dict
):
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"

    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    prompt = config["prompt"]
    
    tokens = model.to_tokens(prompt)
    
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    embed = cache["embed"]
    
    z = cache['z', config["layer_n"]]
    results = t.einsum('shd,hdm->shm', z, model.W_O[config["layer_n"]])

    W_U_correct_tokens = model.W_U[:, tokens.squeeze()[1:]]

    direct_attributions = einops.einsum(W_U_correct_tokens, embed[1:], "emb seq, seq emb -> seq")
    layer_attributions = einops.einsum(W_U_correct_tokens, results[1:], "emb seq, seq nhead emb -> seq nhead")
    
    logit_attr = t.concat([direct_attributions.unsqueeze(-1), layer_attributions], dim=-1)
    
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(model, tokens[1:])
    x_labels = ["Direct"] + [
        f"H{h}" for h in config["head_n"]
    ]
            
    fig = imshow(
        to_numpy(logit_attr[:, tuple([0]+[h+1 for h in config['head_n']])]),
        x=x_labels,
        y=y_labels,
        labels={"x": "Term", "y": "Position", "color": "logit"},
        title="Атрибуция логитов",
        #height=100 + (30 if title else 0) + 15 * len(y_labels),
        #width=24 * len(x_labels),
        #return_fig=True,
    )
    
    content = {
        "fig": fig.to_json()
    }

    return content

def logit_lens(
    data: dict
):
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
    
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    prompt = config["prompt"]
    
    tokens = model.to_tokens(prompt)

    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    res_stream = cache[f'blocks.{config["layer_n"]}.hook_resid_post']
    
    logits = model.unembed(model.ln_final(res_stream))
    last_token_logits = logits[-1]
    
    correct_tokens = tokens.squeeze(0)[1:]
    last_token_logits_correct = last_token_logits[correct_tokens]
        
    fig = imshow(
        to_numpy(last_token_logits_correct.unsqueeze(1)),
        y=convert_tokens_to_string(model, correct_tokens),
        labels={"y": "Position", "color": "logit"},
        title="Логитные линзы",
        #height=100 + (30 if title else 0) + 15 * len(y_labels),
        #width=24 * len(x_labels),
        #return_fig=True,
    )
    
    content = {
        "fig": fig.to_json()
    }

    return content

def activation_patching(
    data: dict
):
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
        
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    prompt_clean = config["correct_prompt"].strip()
    prompt_corrupted = config["corrupted_prompt"].strip()
    answer = config["answer"].strip()
    
    correct_tokens = model.to_tokens(prompt_clean + " " + answer)[:, model.to_tokens(prompt_clean).shape[1]:].squeeze(0)

    tokens_clean = model.to_tokens(prompt_clean)
    clean_logits, cache_clean = model.run_with_cache(tokens_clean, remove_batch_dim=False)

    layer = config["layer_n"]
    head_idx = tuple([h for h in config['head_n']])

    def head_patch_hook(z, hook):
        z = z.clone()
        z[:, :, head_idx, :] = cache_clean["z", layer][:, :, head_idx, :]
        return z

    tokens_corrupted = model.to_tokens(prompt_corrupted)
    patched_logits = model.run_with_hooks(
        tokens_corrupted,
        fwd_hooks=[(utils.get_act_name("z", layer), head_patch_hook)],
    )
        
    logits = t.concat([
        clean_logits.squeeze(0)[-1,correct_tokens].unsqueeze(0),
        patched_logits.squeeze(0)[-1,correct_tokens].unsqueeze(0)
    ], dim=0)
    
    fig = imshow(
        to_numpy(logits),
        y=["Корректный", "Некорректный"],
        x=convert_tokens_to_string(model, correct_tokens),
        labels={"x": "Токен", "y": "Промпт", "color": "logit"},
        title="Атрибуция логитов",
        #height=100 + (30 if title else 0) + 15 * len(y_labels),
        #width=24 * len(x_labels),
        #return_fig=True,
    )
    
    content = {
        "diff": (t.diff(logits, n=1, dim=0)/logits.shape[1])[0][0].tolist(),
        "fig": fig.to_json()
    }

    return content

async def main():
    connection = await aio_pika.connect_robust(
        "amqp://guest:guest@localhost/"
    )

    channel = await connection.channel()

    await channel.set_qos(prefetch_count=1)

    queue = await channel.declare_queue(
        "main_queue",
        durable=True
    )

    await channel.declare_queue(
        "results",
        durable=True
    )

    async with queue.iterator() as q:
        async for message in q:

            async with message.process():

                task = json.loads(message.body)
                
                match task["method"]:
                    case "ablation": 
                        result = ablation(task)
                    case "actmap":
                        result = activation_map(task)
                    case "logattr":
                        result = logit_attribution(task)
                    case "loglens":
                        result = logit_lens(task)
                    case "actpatch":
                        result = activation_patching(task)
                    case _:
                        raise ValueError("Unknown type")

                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(result).encode(),
                        correlation_id=message.correlation_id
                    ),
                    routing_key=message.reply_to
                )

asyncio.run(main())