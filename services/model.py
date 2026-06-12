from contextlib import asynccontextmanager
from fastapi import *
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json
from fastapi.middleware.cors import CORSMiddleware
import psycopg

@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    with psycopg.connect("dbname=modeldb user=user password=123 host=localhost port=5436") as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                title TEXT,
                n_layers INT,
                n_heads INT
            )
            """)
            conn.commit()
            
            cur.execute(f"SELECT * FROM models")
            rows = cur.fetchall()
            if len(rows) == 0:
                cur.execute(f"""
                INSERT INTO models (title,n_layers,n_heads) VALUES ('gpt2-small',12,12)
                """)
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="../templates")
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
    with psycopg.connect("dbname=modeldb user=user password=123 host=localhost port=5436", row_factory=psycopg.rows.dict_row) as conn:
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
    with psycopg.connect("dbname=modeldb user=user password=123 host=localhost port=5436", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT *
            FROM models
            """)
            
            models = cur.fetchall()
    
    return models