from fastapi import *
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
templates = Jinja2Templates(directory="../templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/main")
async def get_file():
    return None

@app.post("/send")
async def send(value: str = Form(...)):
    return None

@app.get("/result")
async def result(request: Request):
    return None