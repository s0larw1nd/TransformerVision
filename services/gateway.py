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
    return FileResponse("../templates/model_choose.html")

@app.post("/send")
async def send(value: str = Form(...)):
    result = value.upper()

    payload = {'model_name': 'gpt2-small'}

    #async with httpx.AsyncClient() as client:
    #    response = await client.get("http://127.0.0.1:81/parse", params=payload, timeout=1000)
    
    redirect = RedirectResponse(url="/result", status_code=status.HTTP_303_SEE_OTHER)
    #redirect.set_cookie(key="layers", value=json.dumps(response.json()['layers']))
    redirect.set_cookie(key="layers", value=json.dumps(['hook_resid_post', 'hook_mlp_out', 'mlp.hook_post', 'mlp.hook_pre', 'ln2.hook_normalized', 'ln2.hook_scale', 
                                                        'hook_resid_mid', 'hook_attn_out', 'attn.hook_z', 'attn.hook_pattern', 'attn.hook_attn_scores', 'attn.hook_v', 
                                                        'attn.hook_k', 'attn.hook_q', 'ln1.hook_normalized', 'ln1.hook_scale', 'hook_resid_pre']))
    return redirect

@app.get("/result")
async def result(request: Request):
    raw = request.cookies.get("layers") 
    data = json.loads(raw) if raw else []
    
    return templates.TemplateResponse(
        request=request,
        name="model.html",   
        context={
            "request": request,
            "data": data
        }
    )