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
from transformer_lens.utils import get_act_name
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
from transformer_lens.model_bridge import TransformerBridge
import struct
import base64
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, required=True)
args = parser.parse_args()

device = t.device(
            "mps"
            if t.backends.mps.is_available()
            else f"cuda:{args.gpu_id}"
            if t.cuda.is_available()
            else "cpu"
        )

def to_numpy(
    tensor: t.Tensor
):
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
        raise ValueError(f"Invalid type: {type(tensor)}")
def convert_tokens_to_string(
    model: HookedTransformer,
    tokens: t.Tensor, 
    batch_index: int = 0
):
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

@t.no_grad()
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
    
    model = HookedTransformer.from_pretrained(model_name, device=device)
    prompt = config["prompt"]
    
    ablation_scores = t.zeros(len(config["head_n"]), device=device)
    tokens = model.to_tokens(prompt)
    
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()
    
    layers = config["layer_n"]
    heads = config["head_n"]
    if len(layers) == 1: layers *= len(heads)

    for i, head_idx in enumerate(heads):
        temp_hook_fn = functools.partial(head_zero_ablation_hook, head_index_to_ablate=head_idx)
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", layers[i]), temp_hook_fn)])
        loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1) :].mean()
    
        ablation_scores[i] = loss - loss_no_ablation
    
    ablation_scores = ablation_scores.unsqueeze(0).transpose(-1,-2).cpu().detach().numpy()
    
    fig = imshow(
        ablation_scores,
        y=[f"{l}.{h}" for l,h in zip(layers, heads)],
        labels={"y": "Голова внимания", "color": "Logit diff"},
        title="Изменения значения функции потерь после аблации",
        text_auto=".2f",
    )

    content = {
        "scores": ablation_scores.squeeze(1).tolist(),
        "fig": fig.to_json()
    }

    return content

@t.no_grad()
def activation_map(
    data: dict  
):
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
    
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

@t.no_grad()
def logit_attribution(
    data: dict
):
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"

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
    )
    
    content = {
        "fig": fig.to_json()
    }

    return content

@t.no_grad()
def logit_lens(
    data: dict
):
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
    
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
    )
    
    content = {
        "fig": fig.to_json()
    }

    return content

@t.no_grad()
def activation_patching(
    data: dict
):
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
        
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
    )
    
    content = {
        "diff": (t.diff(logits, n=1, dim=0)/logits.shape[1])[0][0].tolist(),
        "fig": fig.to_json()
    }

    return content

def linear_probe(
    data: dict
):
    def make_labels(tokens, correct_tokens):
        labels = t.zeros_like(tokens, dtype=t.bool)
        for i in range(tokens.shape[0]):
            labels[i] = t.isin(tokens[i], correct_tokens[i])
        return labels.float()

    def from_jagged(array, pad_value=-1):
        max_len = max(len(row) for row in array)
        padded = [row + [pad_value] * (max_len - len(row)) for row in array]
        return t.tensor(padded, dtype=t.long)

    def find_prompt_tokens_for_string(model, prompt: str, correct: str):
        if not correct:
            return []

        start = prompt.find(correct)
        if start == -1: raise ValueError("ERROR")
        end = start + len(correct)

        encoded = model.tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
            offsets = offsets[0]

        result = []
        for token_id, (tok_start, tok_end) in zip(input_ids, offsets):
            if tok_end > start and tok_start < end:
                result.append(int(token_id))

        return result

    config = json.loads(data['config'])

    prompts = config["prompts"].split("\r\n")

    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
        
    model = HookedTransformer.from_pretrained(model_name, device=device)

    correct_tokens = from_jagged([find_prompt_tokens_for_string(model, pr, cor) for pr,cor in zip(prompts, config["correct"].split("\r\n"))])

    enc = model.tokenizer(prompts, padding=True, return_tensors="pt")
    tokens = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].bool().to(device)

    _, cache = model.run_with_cache(tokens)
    res_stream = cache[f"blocks.{config["layer_n"]}.hook_resid_post"]

    labels = make_labels(tokens, correct_tokens)

    valid = attention_mask
    X = res_stream[valid]
    y = labels[valid].unsqueeze(1)

    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    g = t.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    X_train = t.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    def normalize_batch(x):
        return (x - mean) / std

    probe = t.nn.Linear(X.shape[1], 1).to(device)

    y_train = t.stack([train_dataset[i][1] for i in range(len(train_dataset))]).to(device)
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = (neg / pos).clamp_min(1.0) if pos > 0 else t.tensor(1.0, device=device)

    criterion = t.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = t.optim.Adam(probe.parameters(), lr=1e-3)

    num_epochs = 1000
    for _ in range(num_epochs):
        for xb, yb in train_loader:
            xb = normalize_batch(xb.to(device))
            yb = yb.to(device)

            logits = probe(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    all_preds = []
    all_targets = []
    probe.eval()

    with t.no_grad():
        for xb, yb in test_loader:
            xb = normalize_batch(xb.to(device))
            yb = yb.to(device)
            logits = probe(xb)
            probs = t.sigmoid(logits)
            preds = (probs > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    content = {
        "f1": f1_score(all_targets, all_preds)
    }

    return content

@t.no_grad()
def qk_ov_composition_matrix_bytes(
    data: dict
):
    device = t.device(
            "mps"
            if t.backends.mps.is_available()
            else f"cuda:{args.gpu_id}"
            if t.cuda.is_available()
            else "cpu"
        )
    
    config = json.loads(data['config'])
    
    try:
        model_name = rqsts.get(f"http://localhost:80/model/{data['model_id']}").json()["title"]
    except Exception:
        model_name = "Ошибка"
        
    model = HookedTransformer.from_pretrained(model_name, device=device)

    tokens = model.to_tokens(config["prompt"])
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    L = model.cfg.n_layers
    H = model.cfg.n_heads

    q0 = cache[get_act_name("q", 0, "attn")]
    T = q0.shape[0]
    device = q0.device
    dtype = q0.dtype

    P = L * H
    N = T * P
    out = t.empty((N, N), device=device, dtype=dtype)

    z_proj = [[None] * H for _ in range(L)]
    q_all = [[None] * H for _ in range(L)]

    for l in range(L):
        attn = model.blocks[l].attn
        z_l = cache[get_act_name("z", l, "attn")]
        q_l = cache[get_act_name("q", l, "attn")]

        for h in range(H):
            z_proj[l][h] = z_l[:, h] @ attn.W_O[h]
            q_all[l][h] = q_l[:, h]

    token_ids = t.arange(T, device=device)

    for l1 in range(L):
        for h1 in range(H):
            src = z_proj[l1][h1]
            lh1 = l1 * H + h1

            for l2 in range(L):
                attn2 = model.blocks[l2].attn
                for h2 in range(H):
                    q = q_all[l2][h2]
                    delta_k = src @ attn2.W_K[h2]

                    scores = (q @ delta_k.T) / (
                        q.norm(dim=-1, keepdim=True) * delta_k.norm(dim=-1).unsqueeze(0) + 1e-8
                    )

                    lh2 = l2 * H + h2

                    rows = token_ids * P + lh1
                    cols = token_ids * P + lh2
                    out[rows[:, None], cols[None, :]] = scores

    M = t.where(out > 0.50, out, 0)

    csr = M.to_sparse_csr()

    n = M.shape[0]
    nnz = csr.values().numel()

    header = struct.pack(
        "<ii",
        n,
        nnz
    )

    body = b"".join([
        csr.crow_indices()
            .cpu()
            .numpy()
            .astype("int32")
            .tobytes(),

        csr.col_indices()
            .cpu()
            .numpy()
            .astype("int32")
            .tobytes(),

        csr.values()
            .cpu()
            .numpy()
            .astype("float32")
            .tobytes()
    ])

    content = {
        "tokens": model.to_str_tokens(tokens),
        "tokens_n": T,
        "layers_n": L,
        "heads_n": H,
        "result": base64.b64encode(header + body).decode("ascii")
    }

    return content

async def main():
    connection = await aio_pika.connect_robust(
        "amqp://guest:guest@localhost/"
    )

    channel = await connection.channel(publisher_confirms=True)

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

            async with message.process(requeue=True):

                try:
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
                        case "linprob":
                            result = linear_probe(task)
                        case "composition":
                            result = qk_ov_composition_matrix_bytes(task)
                        case _:
                            raise ValueError("Unknown type")
                except Exception as e:
                    print(f"ERROR: {e}")
                    result = {}

                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(result).encode(),
                        correlation_id=message.correlation_id
                    ),
                    routing_key=message.reply_to
                )

asyncio.run(main())