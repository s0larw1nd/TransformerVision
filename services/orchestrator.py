import asyncio
from contextlib import asynccontextmanager
import datetime
import json
import uuid
from fastapi import *
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.sse import EventSourceResponse
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

rabbit_connection = None
consumer_task = None

results = {}

async def rabbit_consumer(
    connection: aio_pika.AbstractRobustConnection
):
    channel = await connection.channel()
    queue = await channel.declare_queue(
        "results",
        durable=True
    )

    async with queue.iterator() as iterator:
        async for message in iterator:
            async with message.process():
                results[message.correlation_id] = message.body

@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    with psycopg.connect("dbname=orchestratordb user=user password=123 host=localhost port=5434") as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments_runs (
                id SERIAL PRIMARY KEY,
                cell_id INT,
                seq_no INT,
                method_name TEXT,
                config TEXT,
                status TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS experiments_runs_results (
                id SERIAL PRIMARY KEY,
                run_id INT,
                text_json JSONB,
                metrics_json JSONB,
                artifacts_refs_json JSONB
            )
            """)
            conn.commit()

    global rabbit_connection
    rabbit_connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")

    global consumer_task
    consumer_task = asyncio.create_task(rabbit_consumer(rabbit_connection))

    yield

    consumer_task.cancel()
    await consumer_task

    await rabbit_connection.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/parse")
async def parse_config(
    model_name: str
):
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

@app.get("/results/{task_id}")
async def results_event(task_id: str):
    async def generator():
        while True:
            result = results.get(task_id)

            if result is not None:
                data = json.loads(result)
            
                figures_saved = {} 
                if "fig" in data:
                    output_file = Path(f"../stored/{task_id}.json")
                    output_file.parent.mkdir(exist_ok=True, parents=True)

                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(data["fig"])
                    del data["fig"]
                    figures_saved['fig'] = f'{task_id}.json'
                
                with psycopg.connect("dbname=orchestratordb user=user password=123 host=localhost port=5434") as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            UPDATE experiments_runs
                            SET status = 'finished',
                                finished_at = '{datetime.datetime.now(datetime.timezone.utc)}'
                            WHERE id = {task_id}
                        """)
                        
                        cur.execute("""
                            INSERT INTO experiments_runs_results (
                                run_id,
                                text_json,
                                metrics_json,
                                artifacts_refs_json
                            )
                            VALUES (%s,%s,%s,%s)
                        """, (task_id, json.dumps(data) if len(data)!=0 else None, None, json.dumps(figures_saved) if figures_saved else None))
                        
                    conn.commit()

                yield f"data: {result.decode("utf-8")}\n\n"
                del results[task_id]
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(generator())

@app.get("/results")
async def results_test():
    content = {
        "data": list(results.keys())
    }
    return Response(content=json.dumps(content), media_type="application/json")

@app.post("/method")
async def method(
    request: Request, 
    data: dict
):
    channel = await rabbit_connection.channel()

    queue = await channel.declare_queue(
        "main_queue",
        durable=True
    )

    with psycopg.connect("dbname=orchestratordb user=user password=123 host=localhost port=5434") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO experiments_runs (
                    cell_id,
                    seq_no,
                    method_name,
                    config,
                    status,
                    started_at,
                    finished_at
                )
                VALUES (%s,%s,%s,%s,'pending',%s, NULL)
                RETURNING id;
            """, (
                data['cell_id'],
                data['seq_no'],
                data['method'],
                data['config'],
                datetime.datetime.now(datetime.timezone.utc)
            ))
            
            inserted_id = cur.fetchone()[0]
            conn.commit()

    results[inserted_id] = None
    
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(data).encode(),
            correlation_id=inserted_id,
            reply_to="results"
        ),
        routing_key=queue.name
    )

    content = {
        "cid": inserted_id
    }
    
    return Response(content=json.dumps(content), media_type="application/json")

@app.post("/history")
async def get_history(
    request: Request, 
    data: dict
):
    with psycopg.connect("dbname=orchestratordb user=user password=123 host=localhost port=5434", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, method_name, config
                FROM experiments_runs
                WHERE cell_id=%s AND status='finished'
                ORDER BY finished_at DESC
            """, (data['cell_id'],))
            
            history = cur.fetchall()
            
    content = {
        "history": history
    }
    
    return Response(content=json.dumps(content), media_type="application/json")

@app.post("/history/{id}")
async def from_history(
    request: Request, 
    id: int
):
    with psycopg.connect("dbname=orchestratordb user=user password=123 host=localhost port=5434", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT method_name, config
                FROM experiments_runs
                WHERE id=%s
            """, (id,))
            
            method = cur.fetchone()
            
            cur.execute("""
                SELECT * 
                FROM experiments_runs_results
                WHERE run_id=%s
            """, (id,))

            operation = cur.fetchone()
    
    content = {
        'method': method
    }
    
    if operation['artifacts_refs_json']['fig']:
        with open(f"../stored/{operation['artifacts_refs_json']['fig']}", "r", encoding="utf-8") as f:
            content["fig"] = f.read()

    return Response(content=json.dumps(content | operation['text_json']), media_type="application/json")