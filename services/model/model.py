from contextlib import asynccontextmanager
from fastapi import *
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from transformer_lens.model_bridge import TransformerBridge
import torch as t

@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    with psycopg.connect("dbname=modeldb user=user password=123 host=db-model port=5434") as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                title TEXT UNIQUE,
                n_layers INT,
                n_heads INT,
                layers TEXT[]
            )
            """)
            conn.commit()
            
            cur.execute(f"SELECT * FROM models")
            rows = cur.fetchall()
            if len(rows) == 0:
                cur.execute(f"""
                INSERT INTO models (title,n_layers,n_heads,layers) VALUES (%s,%s,%s,%s)
                """, (
                    'gpt2',
                    12,
                    12,
                    ['hook_resid_post', 'hook_mlp_out', 'mlp.hook_post', 'mlp.hook_pre', 'ln2.hook_normalized', 'ln2.hook_scale', 
                    'hook_resid_mid', 'hook_attn_out', 'attn.hook_z', 'attn.hook_pattern', 'attn.hook_attn_scores', 'attn.hook_v', 
                    'attn.hook_k', 'attn.hook_q', 'ln1.hook_normalized', 'ln1.hook_scale', 'hook_resid_pre'][::-1]
                ))

                conn.commit()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
    
@app.get("/model/{id}")
async def get_info(
    id: int
):
    with psycopg.connect("dbname=modeldb user=user password=123 host=db-model port=5434", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT *
            FROM models
            WHERE id = %s
            """, (id,))
            
            model = cur.fetchone()
    
    return model

@app.get("/model/")
async def get_info():
    with psycopg.connect("dbname=modeldb user=user password=123 host=db-model port=5434", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT *
            FROM models
            """)
            
            models = cur.fetchall()
    
    return models

@app.post("/model/new")
async def add_model(
    model_name: str
):
    try:
        device = t.device(
            "mps"
            if t.backends.mps.is_available()
            else "cuda"
            if t.cuda.is_available()
            else "cpu"
        )

        model = TransformerBridge.boot_transformers(model_name, device=device, load_weights=False)

        _, cache = model.run_with_cache(model.to_tokens("Hello world"))

        model_layers = list(cache.keys())[::-1][:-2]

        layers_full: dict[int, list[str]] = {}
        layers = []

        current_block = None
        seen_types = []

        type_map = {
            "mlp": "mlp",
            "attn": "attn",
        }

        for layer in model_layers:
            parts = layer.split(".")

            if len(parts) < 3 or parts[0] != "blocks":
                continue

            block_id = int(parts[1])
            layer_type = parts[2]

            layers_full.setdefault(block_id, []).insert(
                0, ".".join(parts[2:])
            )

            key = (
                type_map.get(layer_type)
                or ("ln" if "ln" in layer_type else None)
            )

            if key is None:
                continue

            if block_id != current_block:
                current_block = block_id
                layers.append([])
                seen_types.clear()

            if not seen_types or seen_types[-1] != key:
                layers[-1].append((block_id, key))
                seen_types.append(key)

        with psycopg.connect("dbname=modeldb user=user password=123 host=db-model port=5434") as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO models (title,n_layers,n_heads,layers) VALUES (%s,%s,%s,%s)
                    """, (
                        model_name,
                        model.cfg.n_layers,
                        model.cfg.n_heads,
                        layers_full[0][::-1]
                    ))
                
                conn.commit()

        return {"status": 200}
    
    except Exception:
        return {"status": 500}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model:app", port=84, log_level="info", reload=True)