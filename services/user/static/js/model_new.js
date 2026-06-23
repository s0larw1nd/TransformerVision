const svg = document.getElementById('modelSvg');
const frame = document.getElementById('modelFrame');
const tooltip = document.getElementById('tooltip');
const modeSwitch = document.getElementById('modeSwitch');
const modeLabel = document.getElementById('modeLabel');
const toolsList = document.getElementById('toolsList');
const historyList = document.getElementById('historyList');
const toolsFields = document.getElementById('toolsFields');
const historyFields = document.getElementById('historyFields');
const resultsArea = document.getElementById('resultsArea');
const promptField = document.getElementById('promptField');
var chosen = null;

modeSwitch.addEventListener("click", () => toggleMode());
document.getElementById("cell-actmap").addEventListener("click", () => renderToolFields('actmap'));
document.getElementById("cell-ablation").addEventListener("click", () => renderToolFields('ablation'));
document.getElementById("cell-logattr").addEventListener("click", () => renderToolFields('logattr'));
document.getElementById("cell-loglens").addEventListener("click", () => renderToolFields('loglens'));
document.getElementById("cell-actpatch").addEventListener("click", () => renderToolFields('actpatch'));
document.getElementById("cell-linprob").addEventListener("click", () => renderToolFields('linprob'));

const state = {
    mode: 'attention',
    selectedToken: null,
    selectedNode: null,
    nodes: [],
    matrices: [],
    history: [],
    toolIndex: 0,
};

function setMode(nextMode) {
    window.location.href = window.location.href + "/struct";
}

function toggleMode() {
    setMode(state.mode === 'attention' ? 'struct' : 'attention');
}

var MODEL = null;

async function getMatrix() {
    const resp = await fetch(`/composition`, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        method: "composition",
        model_id: modelId,
        config: JSON.stringify({ prompt: promptField.value })
    })
    });

    const job = await resp.json();

    const data = await new Promise((resolve, reject) => {
    const source = new EventSource(
        `/composition/${job.cid}`
    );

    source.onmessage = (event) => {
        const temp = JSON.parse(event.data);
        source.close();
        resolve(temp);
    };

    source.onerror = (err) => {
        console.log(err);
        source.close();
        reject(err);
    };
    });

    var matrix = await loadMatrix(job.cid, data);

    return matrix;
}

function nodeId(token, level, head) {
    return `${token}_${level}_${head}`;
}

function nodeFromId(tstr) {
    const t = tstr.split("_");
    return t[0], t[1], t[2]
}

async function loadMatrix(task_id, data){
    const r = await fetch(`/composition/${task_id}/data`);
    const buffer = await r.arrayBuffer();
    const view = new DataView(buffer);

    let offset = 0;
    const n = view.getInt32(offset, true);
    offset += 4;

    const nnz = view.getInt32(offset, true);
    offset += 4;

    const crow = new Int32Array(
    buffer,
    offset,
    n + 1
    );

    offset += (n + 1) * 4;

    const col = new Int32Array(
    buffer,
    offset,
    nnz
    );

    offset += nnz * 4;

    const values = new Float32Array(
    buffer,
    offset,
    nnz
    );

    matrix = {
        n,
        crow,
        col,
        values,
    };
    matrix.tokens_n = data.tokens_n;
    matrix.layers_n = data.layers_n;
    matrix.heads_n = data.heads_n;
    matrix.tokens = data.tokens;

    return matrix;
}

function flatten(token, layer, head, layers, heads) {
    return token * layers * heads
    + layer * heads
    + head;
}

function getAttention(
    matrix,
    token1,
    layer1,
    head1,
    token2,
    layer2,
    head2
){
    const row = flatten(
    token1,
    layer1,
    head1,
    matrix.layers_n,
    matrix.heads_n
    );

    const colTarget = flatten(
    token2,
    layer2,
    head2,
    matrix.layers_n,
    matrix.heads_n
    );

    const start = Number(matrix.crow[row]);
    const end = Number(matrix.crow[row+1]);

    for(let k=start;k<end;k++){
    if(Number(matrix.col[k]) === colTarget)
        return matrix.values[k];
    }

    return 0;
}

async function drawModel() {
    if (!MODEL) {
    var matrix = await getMatrix();
    MODEL = matrix;
    }
    const w = frame.clientWidth;
    const h = frame.clientHeight;

    svg.innerHTML = "";

    const tokenWidth = w / MODEL.tokens.length;

    const top = 70;
    const bottom = 40;

    const levelsHeight = (h - top - bottom) / (MODEL.layers_n + 2);

    const nodes = {};
    let html = "";

    MODEL.tokens.forEach((token,t)=>{
    const x0 = t * tokenWidth;
        
    if(t>0){
        html += `
        <line
        x1="${x0}"
        y1="0"
        x2="${x0}"
        y2="${h}"
        stroke="rgba(0,0,0,.15)"
        />
        `;
    }

    html += `
    <text
    x="${x0+tokenWidth/2}"
    y="25"
    text-anchor="middle"
    font-size="13">
        ${token}
    </text>
    `;

    const embedY = top;

    nodes[nodeId(token,-1,0)] = {
        x:x0+tokenWidth/2,
        y:embedY
    };

    html += square(
        x0+tokenWidth/2,
        embedY,
        12,
        t===state.selectedToken
    );
    
    for(let l=0;l<MODEL.layers_n;l++){
        const y = top + levelsHeight*(l+1);

        for(let head=0;head<MODEL.heads_n;head++){
        const offset = (head - MODEL.heads_n/2) * 12;
        const x = x0+ tokenWidth/2+ offset;
        nodes[nodeId(t,l,head)]={
            x,
            y,
            token:t,
            level:l,
            head
        };

        html += circle(x,y,5,t,l,head,nodeId(t,l,head)==chosen);
        }
    }

    const probY = top + levelsHeight * (MODEL.layers_n+1);

    nodes[nodeId(token,MODEL.layers_n,0)]={
        x:x0+tokenWidth/2,
        y:probY
    };

    html += square(
        x0+tokenWidth/2,
        probY,
        12,
        t===state.selectedToken
    );
    });

    var k = 0;

    const T = MODEL.tokens_n;
    const L = MODEL.layers_n;
    const H = MODEL.heads_n;

    var html_nodes = '';
    var html_chosen = '';

    for (let token1 = 0; token1 < T; token1++) {
    for (let token2 = token1; token2 < T; token2++) {
        for (let layer1 = 0; layer1 < L; layer1++) {
        for (let layer2 = layer1; layer2 < L; layer2++) {
            for (let head1 = 0; head1 < H; head1++) {
            for (let head2 = head1; head2 < H; head2++) {
                const value = getAttention(
                MODEL,
                token1, layer1, head1,
                token2, layer2, head2
                );

                if (value == 0) continue;

                const id1 = nodeId(token1, layer1, head1);
                const id2 = nodeId(token2, layer2, head2);
                const a = nodes[id1];
                const b = nodes[id2];

                if (!a || !b) continue;

                if (chosen === null) {
                html_nodes =
                `<line
                    x1="${a.x}"
                    y1="${a.y}"
                    x2="${b.x}"
                    y2="${b.y}"
                    stroke="LightGray"
                    opacity="0.3"
                    stroke-width="1"
                />` + html_nodes;
                }
                else if ([id1, id2].includes(chosen)) {
                html_chosen = 
                `<line
                    x1="${a.x}"
                    y1="${a.y}"
                    x2="${b.x}"
                    y2="${b.y}"
                    stroke="Black"
                    opacity="1"
                    stroke-width="1"
                />` + html_chosen;
                }
                else {
                html_nodes = 
                `<line
                    x1="${a.x}"
                    y1="${a.y}"
                    x2="${b.x}"
                    y2="${b.y}"
                    stroke="LightGray"
                    opacity="0.1"
                    stroke-width="1"
                />` + html_nodes;
                }

                k += 1;
            }
            }
        }
        }
    }
    }

    html = html_nodes + html_chosen + html

    svg.innerHTML = html;
    bindNodeEvents();
}

promptField.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
    MODEL = null;
    drawModel();
    }
}); 

function circle(x,y,r,token_idx,level_idx,head_idx,chosen){
    return `
    <circle
    cx="${x}"
    cy="${y}"
    r="${r}"
    fill="${chosen ? "black" : "white"}"
    stroke="black"
    class="node"
    data-token=${token_idx}
    data-level=${level_idx}
    data-idx=${head_idx}
    />
    `;
}

function square(x,y,s,selected=false){
    return `
    <rect
    x="${x-s/2}"
    y="${y-s/2}"
    width="${s}"
    height="${s}"
    fill="white"
    stroke="black"
    stroke-width="${selected?3:1}"
    class="node"
    />
    `;
}

function bindNodeEvents() {
    svg.querySelectorAll('.node').forEach(el => {
    el.addEventListener('mouseenter', (e) => {
        var x = 0;
        var y = 0;
        var info = "";
        if (el.tagName == "circle") {
        x = Number(el.cx.baseVal.value);
        y = Number(el.cy.baseVal.value);
        info = `
        слой: ${el.getAttribute("data-level")}<br>
        узел: ${el.getAttribute("data-idx")}<br>
        `;
        }
        else if (el.tagName == "rect") {
        x = Number(el.x.baseVal.value);
        y = Number(el.y.baseVal.value);
        info = `
        `;
        }
        tooltip.innerHTML = info;
        const rect = frame.getBoundingClientRect();
        const left = x + 14;
        const top = y - 6;
        tooltip.style.transform = `translate(${Math.min(left, rect.width - 240)}px, ${Math.max(8, top)}px)`;
    });
    el.addEventListener('mouseleave', () => {
        tooltip.style.transform = 'translate(-9999px, -9999px)';
    });
    el.addEventListener('click', () => {
        if (chosen != nodeId(el.getAttribute("data-token"),el.getAttribute("data-level"),el.getAttribute("data-idx"))) {
        chosen = nodeId(el.getAttribute("data-token"),el.getAttribute("data-level"),el.getAttribute("data-idx"));
        }
        else {
        chosen = null;
        }
        drawModel();

        const head_input = document.getElementById("head-n");
        if (head_input) {
        if (!head_input.value) head_input.value = el.getAttribute("data-idx");
        else head_input.value += `,${el.getAttribute("data-idx")}`;
        }

        const layer_input = document.getElementById("layer-n");
        if (layer_input) {
        if (!layer_input.value) layer_input.value = el.getAttribute("data-level");
        else layer_input.value += `,${el.getAttribute("data-level")}`;
        }
    });
    });
}

function renderToolFields(tool) {
    toolsFields.innerHTML = '';
    var fields = null;
    var func = null;
    switch (tool) {
    case 'actmap':
        fields = {
        names: ["Номер слоя", "Головы внимания"],
        types: ["text", "text"]
        };
        func = actmap;
        break;
    case 'ablation':
        fields = {
        names: ["Номер слоя", "Головы внимания"],
        types: ["text", "text"]
        };
        func = ablation;
        break;
    case 'logattr':
        fields = {
        names: ["Номер слоя", "Головы внимания"],
        types: ["text", "text"]
        };
        func = logattr;
        break;
    case 'loglens':
        fields = {
        names: ["Номер слоя"],
        types: ["text"]
        };
        func = loglens;
        break;
    case 'actpatch':
        fields = {
        names: ["Корректный промпт", "Некорректный промпт", "Корректный прогноз", "Номер слоя", "Головы внимания"],
        types: ["text", "text", "text", "text", "text"]
        };
        func = actpatch;
        break;
    case 'linprob':
        fields = {
        names: ["Номер слоя", "Промпты", "Слова"],
        types: ["text", "file", "file"]
        };
        func = linprob;
        break;
    }
    for (const [i, name] of fields.names.entries()) {
    const type = fields.types[i];
    const wrap = document.createElement('div');
    wrap.className = 'field';
    if (type == "text") wrap.innerHTML = `<label>${name}</label><input id=${translate(name)} type="text" />`;
    else if (type == "file") wrap.innerHTML = `<label>${name}</label><input type="file" accept=".txt" />`;
    toolsFields.appendChild(wrap);
    }

    const btnStart = document.createElement('button');
    btnStart.className = 'btn-link';
    btnStart.innerHTML = `Старт`;
    btnStart.id = 'btnStart';
    btnStart.onclick = function () { func(); };
    toolsFields.appendChild(btnStart);

    const btnCancel = document.createElement('button');
    btnCancel.className = 'btn-link';
    btnCancel.innerHTML = `Отмена`;
    btnCancel.addEventListener('click', (e) => {
    document.getElementById("toolsSubmenu").style.display = 'none';
    document.getElementById("toolsMain").style.display = 'block';
    });
    toolsFields.appendChild(btnCancel);

    document.getElementById("toolsSubmenu").style.display = 'block';
    document.getElementById("toolsMain").style.display = 'none';
    document.getElementById("toolsSubmenu-mini-title").innerHTML = `Параметры инструмента "${translate(tool)}"`;
}

function renderHistory() {
    historyList.innerHTML = '';
    if (!state.history.length) {
    const empty = document.createElement('div');
    empty.className = 'cell muted';
    empty.textContent = 'История пуста';
    historyList.appendChild(empty);
    } else {
    state.history.forEach((item, index) => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.innerHTML = `<span>${translate(item.method_name)}</span><span class="x" data-remove="${item.id}">X</span>`;
        cell.addEventListener('click', (e) => {
        if (e.target && e.target.dataset.remove !== undefined) return;
        renderHistoryFields(item);
        });
        cell.querySelector('.x').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteHistory(item.id);
        });
        historyList.appendChild(cell);
    });
    }
}

async function deleteHistory(id) {
    const resp = await fetch(`/history/${id}/delete`, { method: "DELETE" });
    await get_history();
}

async function clearHistory() {
    state.history.forEach(async (item, _) => {
    await fetch(`/history/${item.id}/delete`, { method: "DELETE" });
    });
    state.history = [];
    renderHistory();
}

function renderHistoryFields(item) {
    historyFields.innerHTML = '';
    const fields = JSON.parse(item.config);
    for (const [key, value] of Object.entries(fields)) {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    wrap.innerHTML = `<label>${translate(key)}</label><textarea readonly>${value}</textarea>`;
    historyFields.appendChild(wrap);
    }
    
    const btnStart = document.createElement('button');
    btnStart.className = 'btn-link';
    btnStart.innerHTML = `Старт`;
    btnStart.onclick = function () { from_history(item.id); };
    historyFields.appendChild(btnStart);

    const btnCancel = document.createElement('button');
    btnCancel.className = 'btn-link';
    btnCancel.innerHTML = `Отмена`;
    btnCancel.addEventListener('click', (e) => {
    document.getElementById("historySubmenu").style.display = 'none';
    document.getElementById("historyMain").style.display = 'block';
    });
    historyFields.appendChild(btnCancel);

    document.getElementById("historySubmenu").style.display = 'block';
    document.getElementById("historyMain").style.display = 'none';
}

document.getElementById('saveBtn').addEventListener('click', () => {
    history.back();
});

document.getElementById('clearHistoryBtn').addEventListener('click', () => {
    clearHistory();
});

function translate(name) {
    switch (name) {
    case 'ablation': return 'Аблация';
    case 'actmap': return 'Карта активаций';
    case 'logattr': return 'Атрибуция логитов';
    case 'loglens': return 'Логитные линзы';
    case 'actpatch': return 'Патчинг активаций';
    case 'linprob': return 'Линейное пробирование';
    
    case 'head_n': return 'Головы внимания';
    case 'layer_n': return 'Слои';
    case 'prompt': return 'Промпт';

    case 'Номер слоя': return 'layer-n';
    case 'Головы внимания': return 'head-n';
    case 'Корректный промпт': return 'correct-prompt';
    case 'Некорректный промпт': return 'incorrect-prompt';
    case 'Корректный прогноз': return 'correct-token';

    default: return name;
    }
}

async function get_data(
    body
) {
    const resp = await fetch(`/method`, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: body
    });

    const job = await resp.json();

    const data = await new Promise((resolve, reject) => {
    const source = new EventSource(
        `/results/${job.cid}`
    );

    source.onmessage = (event) => {
        const temp = JSON.parse(event.data);
        source.close();
        resolve(temp);
    };

    source.onerror = (err) => {
        console.log(err);
        source.close();
        reject(err);
    };
    });

    if (Object.keys(data).length == 0) { throw "ERROR"; }
    return data;
}

function clearSpace() {
    document.getElementById("result-card").innerHTML = '';
    document.getElementById("result-preview").innerHTML = '';
    document.getElementById("result-iframe").srcdoc = '';

    document.getElementById("result-card").style.display = 'none';
    document.getElementById("result-preview").style.display = 'none';
    document.getElementById("result-iframe").style.display = 'none';
}

async function ablation(data = null) {
    var heads = null;
    var layers = null;
    if (data == null) {
    const target = document.getElementById("btnStart");
    
    const items = Array.from(target.parentElement.children);
    const index = items.indexOf(target);

    layers = items[index - 2].lastChild.value
    .replaceAll(/\s*,\s*/g, ",")
    .split(',')
    .map(s => Number(s.trim()));
    heads = items[index - 1].lastChild.value
    .replaceAll(/\s*,\s*/g, ",")
    .split(',')
    .map(s => Number(s.trim()));

    data = await get_data(
        JSON.stringify({
        method: "ablation",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ head_n: heads, layer_n: layers, prompt: promptField.value })
        })
    );
    }
    else {
    const config = JSON.parse(data.method.config);
    heads = config.head_n;
    layers = config.layer_n;
    };

    fig = JSON.parse(data.fig);

    clearSpace();
    document.getElementById("result-empty").style.display = 'none';
    document.getElementById("result-card").style.display = 'block';

    var res = '<p>Изменения функции потерь:<\p>'
    if (layers.length == 1) {
    heads.forEach((head, index) => {
        res += `<p>Г.В. ${head}: ${data.scores[index]}<\p>`
    });
    }
    else {
    heads.forEach((head, index) => {
        console.log()
        res += `<p>Г.В. ${layers[index]}.${head}: ${data.scores[index]}<\p>`
    });
    }
    
    document.getElementById("result-card").innerHTML = res;
    document.getElementById("result-preview").style.display = 'block';
    Plotly.newPlot("result-preview", fig.data, fig.layout);

    get_history();
}

async function actmap(data = null) {
    var heads = null;
    if (data == null) {
    const target = document.getElementById("btnStart");
    
    const items = Array.from(target.parentElement.children);
    const index = items.indexOf(target);

    const layer = parseInt(items[index - 2].lastChild.value);
    heads = items[index - 1].lastChild.value
    .split(',')
    .map(s => Number(s.trim()));

    data = await get_data(
        JSON.stringify({
        method: "actmap",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ head_n: heads, layer_n: layer, prompt: promptField.value })
        })
    );
    };

    clearSpace();
    document.getElementById("result-empty").style.display = 'none';
    document.getElementById("result-card").style.display = 'block';
    var res = `<p>Паттерны:<\p>
    <p>Текущий токен: ${(data.current.length!=0) ? data.current.toString() : 'Не найдены'}<\p>
    <p>Предыдущий токен: ${(data.prev.length!=0) ? data.prev.toString() : 'Не найдены'}<\p>
    <p>Первый токен: ${(data.first.length!=0) ? data.first.toString() : 'Не найдены'}<\p>
    <p>Индукция: ${(data.induction.length!=0) ? data.induction.toString() : 'Не найдены'}<\p>
    `;
    document.getElementById("result-card").innerHTML = res;

    document.getElementById("result-iframe").style.display = 'block';
    document.getElementById("result-iframe").srcdoc = data.fig;

    get_history();
}

async function logattr(data = null) {
    var heads = null;
    if (data == null) {
    const target = document.getElementById("btnStart");
    
    const items = Array.from(target.parentElement.children);
    const index = items.indexOf(target);

    const layer = parseInt(items[index - 2].lastChild.value);
    heads = items[index - 1].lastChild.value
    .split(',')
    .map(s => Number(s.trim()));

    data = await get_data(
        JSON.stringify({
        method: "logattr",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ head_n: heads, layer_n: layer, prompt: promptField.value })
        })
    );
    };
    
    fig = JSON.parse(data.fig);

    clearSpace();
    document.getElementById("result-empty").style.display = 'none';
    document.getElementById("result-preview").style.display = 'block';
    Plotly.newPlot("result-preview", fig.data, fig.layout);

    get_history();
}

async function loglens(data = null) {
    if (data == null) {
    const target = document.getElementById("btnStart");
    
    const items = Array.from(target.parentElement.children);
    const index = items.indexOf(target);

    const layer = parseInt(items[index - 1].lastChild.value);

    data = await get_data(
        JSON.stringify({
        method: "loglens",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ layer_n: layer, prompt: promptField.value })
        })
    );
    }
    
    fig = JSON.parse(data.fig);

    clearSpace();
    document.getElementById("result-empty").style.display = 'none';
    document.getElementById("result-preview").style.display = 'block';
    Plotly.newPlot("result-preview", fig.data, fig.layout);

    get_history();
}

async function actpatch(data = null) {
    var heads = null;
    if (data == null) {
    const target = document.getElementById("btnStart");
    
    const items = Array.from(target.parentElement.children);
    const index = items.indexOf(target);
    
    const correct_prompt = items[index - 5].lastChild.value;
    const corrupted_prompt = items[index - 4].lastChild.value;
    const answer = items[index - 3].lastChild.value;
    const layer = parseInt(items[index - 2].lastChild.value);
    heads = items[index - 1].lastChild.value
    .split(',')
    .map(s => Number(s.trim()));

    data = await get_data(
        JSON.stringify({
        method: "actpatch",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ 
            head_n: heads, 
            layer_n: layer, 
            correct_prompt: correct_prompt,
            corrupted_prompt: corrupted_prompt,
            answer: answer
        })
        })
    );
    }
    
    fig = JSON.parse(data.fig);

    clearSpace();
    document.getElementById("result-empty").style.display = 'none';
    document.getElementById("result-card").style.display = 'block';
    var res = `Среднее изменение: ${data.diff}`;
    document.getElementById("result-card").innerHTML = res;

    document.getElementById("result-preview").style.display = 'block';
    Plotly.newPlot("result-preview", fig.data, fig.layout);

    get_history();
}

async function linprob(data = null) {
    if (data == null) {
    const target = document.getElementById("btnStart");
    
    const items = Array.from(target.parentElement.children);
    const index = items.indexOf(target);

    const layer = parseInt(items[index - 3].lastChild.value);
    const prompts = await items[index - 2].lastChild.files[0].text();
    const correct = await items[index - 1].lastChild.files[0].text();

    data = await get_data(
        JSON.stringify({
        method: "linprob",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({
            layer_n: layer,
            prompts: prompts,
            correct: correct
        })
        })
    );
    }

    clearSpace();
    var res = `F1 score: ${data.f1}`;
    document.getElementById("result-empty").style.display = 'none';
    document.getElementById("result-card").style.display = 'block';
    document.getElementById("result-card").innerHTML = res;
    get_history();
}

async function get_history() {
    const resp = await fetch(`/history`, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        cell_id: cellId
        })
    });

    data = await resp.json();
    state.history = data.history;
    renderHistory();
}

async function from_history(id) {
    const resp = await fetch(`/history/${id}`, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({ })
    });
    
    data = await resp.json();

    switch (data.method.method_name) {
    case 'ablation': ablation(data); break;
    case 'actmap': actmap(data); break;
    case 'logattr': logattr(data); break;
    case 'loglens': loglens(data); break;
    case 'actpatch': actpatch(data); break;
    case 'linprob': linprob(data); break;
    }
}

get_history();
const ro = new ResizeObserver(() => drawModel());
ro.observe(frame);