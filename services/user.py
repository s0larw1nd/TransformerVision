import datetime
from typing import Annotated, List
from fastapi import *
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import jwt
import psycopg
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import requests

load_dotenv("../secret.env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    with psycopg.connect("dbname=userdb user=user password=123 host=localhost port=5435") as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                title TEXT,
                description TEXT,
                model_id TEXT,
                owner_id INT,
                status TEXT,
                is_private BOOLEAN,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                title TEXT,
                description TEXT,
                project_id INT,
                created_by INT,
                created_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS cells (
                id SERIAL PRIMARY KEY,
                experiment_id INT,
                pos INT,
                type TEXT,
                stext TEXT
            );
            
            CREATE TABLE IF NOT EXISTS projects_users (
                id SERIAL PRIMARY KEY,
                user_id INT,
                project_id INT  
            )
            """)
            conn.commit()
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="../templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)  

@app.get("/profile")
async def profile(request: Request):  
    token = request.cookies.get("access_token")
    
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        idx = int(payload.get("sub"))
        if idx is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception

    projects = []
    
    with psycopg.connect(
        "dbname=userdb user=user password=123 host=localhost port=5435",
        row_factory=psycopg.rows.namedtuple_row
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, description, model_id, owner_id, status, is_private, created_at, updated_at
                FROM projects
                WHERE owner_id = %s
                ORDER BY updated_at DESC
                """, (idx,))
            
            projs = cur.fetchall()
            
            for proj in projs:
                cur.execute("""
                SELECT user_id
                FROM projects_users
                WHERE project_id = %s
                """, (proj.id,))
                
                try:
                    model_name = requests.get(f"http://localhost:80/model/{proj.model_id}").json()["title"]
                except Exception:
                    model_name = "Ошибка"
                
                members = []
                try:
                    for m in cur.fetchall():
                        members.append(requests.get(f"http://localhost:80/user/{m.user_id}").json()["username"])
                except Exception:
                    members = []
                
                projects.append(
                    {
                        "id": proj.id,
                        "title": proj.title,
                        "description": proj.description,
                        "model_id": model_name,
                        "owner_id": proj.owner_id,
                        "status": proj.status,
                        "is_private": proj.is_private,
                        "created_at": proj.created_at,
                        "updated_at": proj.updated_at,
                        "members": members
                    }
                )
    
    username = requests.get(f"http://localhost:80/user/{idx}").json()["username"]
    
    context = {
        "username": username,
        "projects": projects
    }

    return templates.TemplateResponse(request, name="profile.html", context=context)

@app.get("/project/create")
async def create_project(request: Request):
    models = requests.get(f"http://localhost:80/model/").json()
    context = {
        "models": models
    }
    return templates.TemplateResponse(
        request=request, name="create_project.html", context=context
    )
@app.post("/project/create")
async def create_project(
    request: Request,
    title: Annotated[str, Form()],
    description: Annotated[str, Form()],
    model: Annotated[int, Form()],
    is_private: Annotated[bool | None, Form()] = False
    ):
    
    token = request.cookies.get("access_token")
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        idx = int(payload.get("sub"))
        if idx is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
    
    with psycopg.connect(
        "dbname=userdb user=user password=123 host=localhost port=5435",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO projects (title,description,model_id,owner_id,status,is_private,created_at,updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (title, description, model, idx, "Active", is_private, datetime.datetime.now(datetime.timezone.utc), datetime.datetime.now(datetime.timezone.utc)))
            inserted_id = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO projects_users (user_id,project_id) VALUES (%s,%s);
            """, (idx, inserted_id))
            conn.commit()
            
    return RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/project/{id}")
async def project(request: Request, id: int):
    with psycopg.connect(
        "dbname=userdb user=user password=123 host=localhost port=5435",
        row_factory=psycopg.rows.namedtuple_row
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, description, model_id, owner_id, status, is_private, created_at, updated_at
                FROM projects
                WHERE id = %s
                ORDER BY updated_at DESC
                """, (id,))
            
            proj_info = cur.fetchone()
            
            cur.execute("""
            SELECT user_id
            FROM projects_users
            WHERE project_id = %s
            """, (id,))
            
            try:
                model_name = requests.get(f"http://localhost:80/model/{proj_info.model_id}").json()["title"]
            except Exception:
                model_name = "Ошибка"
            
            members_ids = cur.fetchall()
            members = []
            
            try:
                for m in members_ids:
                    members.append(requests.get(f"http://localhost:80/user/{m.user_id}").json()["username"])
            except Exception:
                members = []
            
            context = {
                "project": {
                    "id": proj_info.id,
                    "title": proj_info.title,
                    "description": proj_info.description,
                    "model_id": model_name,
                    "owner_id": proj_info.owner_id,
                    "status": proj_info.status,
                    "is_private": proj_info.is_private,
                    "created_at": proj_info.created_at,
                    "updated_at": proj_info.updated_at,
                    "members": members
                }
            }
            
            cur.execute("""
            SELECT *
            FROM experiments
            WHERE project_id = %s
            """, (id,))
            
            experiments = []
            for exp in cur.fetchall():
                try:
                    created_by = requests.get(f"http://localhost:80/user/{exp.created_by}").json()["username"]
                except Exception:
                    created_by = exp.created_by
                    
                experiments.append({
                    "id": exp.id,
                    "title": exp.title,
                    "description": exp.description,
                    "created_by": created_by,
                    "created_at": exp.created_at
                })
            
            context['experiments'] = experiments
    
    return templates.TemplateResponse(
        request=request, name="project_info.html", context=context
    )
    
@app.get("/project/{id_proj}/experiment/{id_exp}")
async def project(request: Request, id_proj: int, id_exp: int):
    with psycopg.connect("dbname=userdb user=user password=123 host=localhost port=5435", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT *
            FROM cells
            WHERE experiment_id = %s
            ORDER BY pos ASC
            """, (id_exp,))
            
            cells = cur.fetchall()
            
            cur.execute("""
            SELECT *
            FROM experiments
            WHERE id = %s
            """, (id_exp,))
            
            exp = cur.fetchone()

    context = {
        "experiment": exp,
        "cells": cells
    }
    
    return templates.TemplateResponse(
        request=request, name="notebook.html", context=context
    )

class Cell(BaseModel):
    id: int
    pos: int
    type: str
    stext: str | None = None

@app.post("/project/{id_proj}/experiment/{id_exp}/sync")
async def project_cells_sync(request: Request, id_proj: int, id_exp: int, cells: List[Cell]):
    with psycopg.connect("dbname=userdb user=user password=123 host=localhost port=5435") as conn:
        with conn.cursor() as cur:
            for cell in cells:
                cur.execute("""
                    INSERT INTO cells (id,experiment_id,pos,type,stext)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        pos = EXCLUDED.pos,
                        stext = EXCLUDED.stext
                """, (cell.id, id_exp, cell.pos, cell.type, cell.stext))

        conn.commit()
        
@app.get("/project/{id_proj}/experiment/{id_exp}/cell/{id_cell}/model")
async def project_cell_model(request: Request, id_proj: int, id_exp: int, id_cell: int):
    with psycopg.connect("dbname=userdb user=user password=123 host=localhost port=5435") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_id
                FROM projects
                WHERE id=%s
            """, (id_proj,))
            
            model_id = cur.fetchone()[0]
            
    context = {
        "experiment": {
            "id": id_exp,
        },
        "cell": {
            "id": id_cell,
        },
        "model": {
            "id": model_id,
            "n_head": 12,
            "n_layers": 12
        },
        "layers": ['hook_resid_post', 'hook_mlp_out', 'mlp.hook_post', 'mlp.hook_pre', 'ln2.hook_normalized', 'ln2.hook_scale', 
                    'hook_resid_mid', 'hook_attn_out', 'attn.hook_z', 'attn.hook_pattern', 'attn.hook_attn_scores', 'attn.hook_v', 
                    'attn.hook_k', 'attn.hook_q', 'ln1.hook_normalized', 'ln1.hook_scale', 'hook_resid_pre'][::-1],
        "op_history": [],
        "metrics": [],
    }
    
    return templates.TemplateResponse(
        request=request, name="model.html", context=context
    )