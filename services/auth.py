from contextlib import asynccontextmanager
import datetime
import os
import time
from dotenv import load_dotenv
from fastapi import *
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
import psycopg
from pydantic import BaseModel
from typing import Annotated

load_dotenv("../secret.env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    with psycopg.connect("dbname=authdb user=user password=123 host=localhost port=5433") as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE,
                password TEXT
            );
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
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY")

class User(BaseModel):
    idx: int
    username: str
    password: str

def check_password(
    psw: str
):
    return len(psw) > 3 and len(psw) < 64

def get_user(
    idx: int = None, 
    username: str = None, 
    password: str = None
):
    if idx:
        q = f"SELECT * FROM users WHERE id={idx}"
    elif username:
        q = f"SELECT * FROM users WHERE username='{username}'"
        
    q += f" AND password='{password}';" if password else ";"
    
    with psycopg.connect("dbname=authdb user=user password=123 host=localhost port=5433", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
            
            print(rows, rows[0])
            if len(rows) == 1: return User(idx=rows[0]['id'], username=rows[0]['username'], password=rows[0]['password'])
            return None

@app.get("/")
async def get_file():
    return FileResponse("../templates/registration.html")
    
@app.post("/register")
async def register(
    username: Annotated[str, Form()], 
    password: Annotated[str, Form()]
):
    if check_password(password):
        with psycopg.connect("dbname=authdb user=user password=123 host=localhost port=5433") as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO users (username, password) VALUES ('{username}', '{password}')"
                )
                conn.commit()
        return FileResponse("../templates/registration_success.html")
    else:
        return FileResponse("../templates/registration_failure.html")

@app.get("/login")
async def get_file():
    return FileResponse("../templates/login.html")
@app.post("/login")
async def login(
    username: Annotated[str, Form()], 
    password: Annotated[str, Form()]
):
    user = get_user(username=username, password=password)
    if not(user): return {"code": "Incorrect password"}

    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    
    token = jwt.encode(
        {"sub": str(user.idx), "exp": expire},
        'secret',
        algorithm="HS256"
    )

    response = RedirectResponse(url="/profile", status_code=303)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax"
    )
    return response

@app.get("/logout")
async def logout(
    request: Request
):
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("access_token")
    return response

@app.get("/user/{id}")
async def get_info(
    id: int
):
    with psycopg.connect("dbname=authdb user=user password=123 host=localhost port=5433", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT username
            FROM users
            WHERE id = %s
            """, (id,))
            
            user = cur.fetchone()
    
    return user

async def get_current_user(
    token: Annotated[str, Depends(OAuth2PasswordBearer(tokenUrl="token"))]
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        idx = int(payload.get("sub"))
        if idx is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
    user = get_user(idx=idx)
    if user is None:
        raise credentials_exception
    return user