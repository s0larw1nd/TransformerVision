from contextlib import asynccontextmanager
import datetime
import os
from fastapi import *
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import jwt
import psycopg
from pydantic import BaseModel
from typing import Annotated

@asynccontextmanager
async def lifespan(app: FastAPI):
    with psycopg.connect("dbname=authdb user=user password=123 host=db-auth port=5432") as conn:
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
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
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
    
    with psycopg.connect("dbname=authdb user=user password=123 host=db-auth port=5432", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
            
            if len(rows) == 1: return User(idx=rows[0]['id'], username=rows[0]['username'], password=rows[0]['password'])
            return None

@app.get("/")
async def get_file(
    request: Request
):
    try:
        token = request.cookies.get("access_token")
        if not token: raise Exception

        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        idx = int(payload.get("sub"))
        if idx is None: raise Exception
    except Exception:
        return templates.TemplateResponse(request, "registration.html")
    else:
        return RedirectResponse(url="/profile", status_code=303)
    
@app.post("/register")
async def register(
    request: Request,
    username: Annotated[str, Form()], 
    password: Annotated[str, Form()]
):
    if not check_password(password):
        return templates.TemplateResponse(
            request,
            "registration.html",
            {
                "message": "Пароль не соответствует требованиям"
            }
        )
    
    try:
        with psycopg.connect("dbname=authdb user=user password=123 host=db-auth port=5432") as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, password)
                    VALUES (%s, %s)
                    """, (username, password))
                conn.commit()
        return templates.TemplateResponse(request, "registration_success.html")
    except psycopg.errors.UniqueViolation:
        return templates.TemplateResponse(
            request,
            "registration.html",
            {
                "message": "Такой логин уже существует"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "registration.html",
            {
                "message": f"Ошибка регистрации: {e}"
            }
        )

@app.get("/login")
async def get_file(
    request: Request
):
    try:
        token = request.cookies.get("access_token")
        if not token: raise Exception

        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        idx = int(payload.get("sub"))
        if idx is None: raise Exception
    except Exception:
        return templates.TemplateResponse(request, "login.html")
    else:
        return RedirectResponse(url="/profile", status_code=303)
@app.post("/login")
async def login(
    request: Request,
    username: Annotated[str, Form()], 
    password: Annotated[str, Form()]
):
    user = get_user(username=username, password=password)
    if not(user):
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "message": f"Ошибка входа: Некорректный логин или пароль"
            }
        )

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
    with psycopg.connect("dbname=authdb user=user password=123 host=db-auth port=5432", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT username
            FROM users
            WHERE id = %s
            """, (id,))
            
            user = cur.fetchone()
    
    return user


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("auth:app", port=82, log_level="info", reload=True)