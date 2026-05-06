from contextlib import asynccontextmanager
import datetime
import json
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

class ExperimentRequest(BaseModel):
    experiment_id: int
    model_id: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    with psycopg.connect("dbname=orchestratordb user=user password=123 host=localhost port=5434") as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments_runs (
                id SERIAL PRIMARY KEY,
                cell_id INT,
                seq_no INT,
                method_name TEXT,
                hook_name TEXT,
                config TEXT,
                status TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                runtime INT
            );
            
            CREATE TABLE IF NOT EXISTS experiments_runs_results (
                id SERIAL PRIMARY KEY,
                run_id INT,
                metrics_json JSONB,
                artifacts_refs_json JSONB
            )
            """)
            conn.commit()
            
            cur.execute(f"SELECT * FROM experiments_runs")
            rows = cur.fetchall()
            if len(rows) == 0:
                data = {
                    "type": "aboba"
                }
                
                cur.execute("""
                INSERT INTO experiments_runs (cell_id,seq_no,method_name,hook_name,config,status,started_at,finished_at,runtime) 
                VALUES (2, 1, 'Ablation', 'blocks.3.hook_resid_pre', %s, 'Success', %s, %s, 0)            
                """, (json.dumps(data), datetime.datetime.now(datetime.timezone.utc), datetime.datetime.now(datetime.timezone.utc)))
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_numpy(tensor):
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
def plot_logit_attribution(
    model, logit_attr: t.Tensor, tokens: t.Tensor, title: str = "", filename: str | None = None
):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(model, tokens[:-1])
    x_labels = ["Direct"] + [
        f"H{h}" for h in range(model.cfg.n_heads)
    ]
    
    fig = imshow(
        to_numpy(logit_attr),  # type: ignore
        x=x_labels,
        y=y_labels,
        labels={"x": "Term", "y": "Position", "color": "logit"},
        title=title if title else None,
        #height=100 + (30 if title else 0) + 15 * len(y_labels),
        #width=24 * len(x_labels),
        #return_fig=True,
    )
    #fig.show()

    return fig

def logdb(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        print(f"[LOG] Вызов {func.__name__} с args={args}, kwargs={kwargs}")
        start = datetime.datetime.now(datetime.timezone.utc)
        
        with psycopg.connect("dbname=orchestratordb user=user password=123 host=localhost port=5434") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO experiments_runs (
                        experiment_id,
                        model_id,
                        method_name,
                        status,
                        started_at,
                        finished_at,
                        runtime
                    )
                    VALUES (%s, %s, %s, %s, %s, NULL, NULL)
                    RETURNING id;
                """, (
                    kwargs['request'].experiment_id,
                    kwargs['request'].model_id,
                    func.__name__,
                    'pending',
                    start
                ))
                
                inserted_id = cur.fetchone()[0]
                conn.commit()
                
        result = await func(*args, **kwargs) 
        
        finish = datetime.datetime.now(datetime.timezone.utc)
        with psycopg.connect("dbname=orchestratordb user=user password=123 host=localhost port=5434") as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE experiments_runs
                    SET status = 'finished',
                        finished_at = '{finish}',
                        runtime = '{int((finish-start).total_seconds() * 1000)}'
                    WHERE id = {inserted_id}
                """)
                
                conn.commit()
        
        return result
    
    return wrapper

@app.get("/parse")
async def parse_config(model_name):
    layers = []
    layers_full = {}
    
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    tokens = model.to_tokens("Hello world")
    logits, cache = model.run_with_cache(tokens)
    
    model_layers = list(cache.keys())[::-1][:-2]
    
    current_block = None
    seen_types = []

    for layer in model_layers:
        parts = layer.split(".")

        if parts[0] != "blocks":
            continue
        if len(parts) < 3:
            continue

        block_id = int(parts[1])
        if block_id not in layers_full: layers_full[block_id] = [] 
        layers_full[block_id].insert(0, ".".join(parts[2:]))
        layer_type = parts[2]

        if block_id != current_block:
            current_block = block_id
            layers.append([])
            seen_types = []
        
        if layer_type == "mlp":
            key = "mlp"
        elif layer_type == "attn":
            key = "attn"
        elif "ln" in layer_type:
            key = "ln"
        else:
            continue
        
        if not(seen_types) or seen_types[-1] != key:
            layers[-1].append((block_id, key))
            seen_types.append(key)
    
    layer_full = layers_full[0][::-1]
    
    return {"layers": layer_full}

@app.post("/ablation")
@logdb
async def ablation(request: ExperimentRequest):
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
        
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    prompt = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    
    ablation_scores = t.zeros(model.cfg.n_heads, device=device)
    tokens = model.to_tokens(prompt)
    
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()
    
    for head in range(model.cfg.n_heads):
        temp_hook_fn = functools.partial(head_zero_ablation_hook, head_index_to_ablate=head)
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", 1), temp_hook_fn)])
        loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1) :].mean()
        ablation_scores[head] = loss - loss_no_ablation
    
    ablation_scores = ablation_scores.unsqueeze(0).transpose(-1,-2).cpu().detach().numpy()
    fig = imshow(
        ablation_scores,
        labels={"y": "Голова внимания", "color": "Logit diff"},
        title="Изменения значения функции потерь после аблации",
        text_auto=".2f",
    )
    return Response(content=fig.to_json(), media_type="application/json")

@app.post("/actmap")
@logdb
async def activation_map(request: ExperimentRequest):
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    prompt = "Hello world"
    
    gpt2_tokens = model.to_tokens(prompt)
    gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
            
    attention_pattern = gpt2_cache["pattern", 1]
    gpt2_str_tokens = model.to_str_tokens(prompt)

    html = cv.attention.attention_patterns(
        gpt2_str_tokens,
        attention_pattern
    )
    
    current_token = []
    prev_token = []
    first_token = []
    induction = []
    for head in range(model.cfg.n_heads):
        attention_pattern = gpt2_cache["pattern", 1][head]
        score_current = attention_pattern.diagonal().mean()
        score_prev = attention_pattern.diagonal(-1).mean()
        score_first = attention_pattern[:, 0].mean()
        
        seq_len = (attention_pattern.shape[-1] - 1) // 2
        score_induction = attention_pattern.diagonal(-seq_len + 1).mean()
        if score_current > 0.4:
            current_token.append(str(head))
        if score_prev > 0.4:
            prev_token.append(str(head))
        if score_first > 0.4:
            first_token.append(str(head))
        if score_induction > 0.4:
            induction.append(str(head))

    return JSONResponse({"html": html._repr_html_(), "current": current_token, "prev": prev_token, "first": first_token, "induction": induction})

@app.post("/logattr")
@logdb
async def logit_attribution(request: ExperimentRequest):
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    prompt = "Hello world"
    
    gpt2_tokens = model.to_tokens(prompt)

    gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)

    embed = gpt2_cache["embed"]
    
    z = gpt2_cache['z', 1]
    results = t.einsum('shd,hdm->shm', z, model.W_O[1])

    W_U_correct_tokens = model.W_U[:, gpt2_tokens.squeeze()[1:]]

    direct_attributions = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
    layer_attributions = einops.einsum(W_U_correct_tokens, results[:-1], "emb seq, seq nhead emb -> seq nhead")
    
    logit_attr = t.concat([direct_attributions.unsqueeze(-1), layer_attributions], dim=-1)
    
    fig = plot_logit_attribution(model, logit_attr, gpt2_tokens, title="Атрибуция логитов")
    
    return Response(content=fig.to_json(), media_type="application/json")

@app.post("/loglens")
@logdb
async def logit_lens(request: ExperimentRequest):
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    prompt = "Hello world"
    
    gpt2_tokens = model.to_tokens(prompt)

    gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
    
    res_stream = gpt2_cache['blocks.8.hook_resid_post']
    
    logits = model.unembed(model.ln_final(res_stream))
    last_token_logits = logits[-1]
    
    print(logits.shape)
    
@app.post("/actpatch")
@logdb
async def activation_patching(request: ExperimentRequest):
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    prompt_clean = "The capital of France is"
    prompt_corrupted = "The capital of Germany is"

    tokens_clean = model.to_tokens(prompt_clean)
    _, cache_clean = model.run_with_cache(tokens_clean, remove_batch_dim=False)

    layer = 5
    head_idx = 5

    def head_patch_hook(z, hook):
        z = z.clone()
        z[:, :, head_idx, :] = cache_clean["z", layer][:, :, head_idx, :]
        return z

    tokens_corrupted = model.to_tokens(prompt_corrupted)
    patched_logits = model.run_with_hooks(
        tokens_corrupted,
        fwd_hooks=[(utils.get_act_name("z", layer), head_patch_hook)],
    )
    
    print(patched_logits.shape)