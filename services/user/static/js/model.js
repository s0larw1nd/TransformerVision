document.getElementById("btn-save").addEventListener("click", () => save());
document.getElementById("btn-reset").addEventListener("click", () => save());

document.getElementById("actmap").addEventListener("click", () => show('actmap'));
document.getElementById("ablation").addEventListener("click", () => show('ablation'));
document.getElementById("logattr").addEventListener("click", () => show('logattr'));
document.getElementById("loglens").addEventListener("click", () => show('loglens'));
document.getElementById("actpatch").addEventListener("click", () => show('actpatch'));
document.getElementById("linprob").addEventListener("click", () => show('linprob'));
document.getElementById("sae").addEventListener("click", () => show('sae'));
document.getElementById("funcvec").addEventListener("click", () => show('funcvec'));

document.getElementById("btn-actmap").addEventListener("click", () => actmap());
document.getElementById("btn-ablation").addEventListener("click", () => ablation());
document.getElementById("btn-logattr").addEventListener("click", () => logattr());
document.getElementById("btn-loglens").addEventListener("click", () => loglens());
document.getElementById("btn-actpatch").addEventListener("click", () => actpatch());
document.getElementById("btn-linprob").addEventListener("click", () => linprob());
document.getElementById("btn-sae").addEventListener("click", () => sae());
document.getElementById("btn-funcvec").addEventListener("click", () => funcvec());

document.getElementById("scope-general").addEventListener("change", () => change_scope());

const layer = document.getElementById("scope-layer");
const head = document.getElementById("scope-head");

let heads;

change_scope();
get_history();

function save() {
    history.back();
}

function change_scope() {
    var general = document.getElementById("scope-general").value;

    layer.innerHTML = '';
    head.innerHTML = '';

    const diagram = document.getElementsByClassName("model-diagram")[0];
    diagram.innerHTML = '';
    if (general == "Модель") {
        layer.style.display = 'none';
        head.style.display = 'none';

        var html = `
        <div class="node">Входные токены</div>
        <div class="arrow">↓</div>
        <div class="node">Эмбеддинги</div>
        <div class="arrow">↓</div>`

        for (let i = 0; i < model_layers; i++) {
            html += `
                <div class="node" data-value="${i+1}">Слой ${i+1} · Attention + MLP</div>
                <div class="arrow">↓</div>
            `;
        }

        html += `<div class="node">Layernorm</div>
        <div class="arrow">↓</div>
        <div class="node">Выходные логиты</div>
        <div class="arrow">↓</div>
        <div class="node">Softmax</div>
        <div class="arrow">↓</div>
        <div class="node">Выходные вероятности</div>
        `;
        
        diagram.innerHTML = html;

        document.getElementById("actmap-slot").style.display = 'none';
        document.getElementById("ablation-slot").style.display = 'none';
        document.getElementById("logattr-slot").style.display = 'none';
        document.getElementById("loglens-slot").style.display = 'none';
        document.getElementById("actpatch-slot").style.display = 'block';
        document.getElementById("linprob-slot").style.display = 'none';
        document.getElementById("sae-slot").style.display = 'none';
        document.getElementById("funcvec-slot").style.display = 'none';
    }
    else if (general == "Слой") {
        layer.style.display = 'block';
        head.style.display = 'none';
        var html = ``;

        for (let i = 0; i < model_layers; i++) {
            layer.add(new Option(i, i));

            html += `
                <div class="node" data-value="${i}">${i}</div>
            `;
            if (i != model_layers-1) html += `<div class="arrow">↓</div>`;
        }

        diagram.innerHTML = html;

        document.getElementById("actmap-slot").style.display = 'block';
        document.getElementById("ablation-slot").style.display = 'block';
        document.getElementById("logattr-slot").style.display = 'block';
        document.getElementById("loglens-slot").style.display = 'block';
        document.getElementById("actpatch-slot").style.display = 'none';
        document.getElementById("linprob-slot").style.display = 'none';
        document.getElementById("sae-slot").style.display = 'none';
        document.getElementById("funcvec-slot").style.display = 'none';
    }
}

function show(id) {
    const submenu = document.getElementById(id).nextElementSibling;
    submenu.classList.toggle('active');
}

async function get_data(
    body
) {
    const resp = await fetch(`http://nginx:80/method`, {
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

    return data;
}

async function ablation(data = null) {
    if (data == null) {
    const input = document.getElementById('ablation-heads');
    heads = input.value
    .split(',')
    .map(s => Number(s.trim()));

    data = await get_data(
        JSON.stringify({
        method: "ablation",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ head_n: heads, layer_n: parseInt(layer.value), prompt: document.getElementById('prompt-field').value })
        })
    );
    }
    else {
    const config = JSON.parse(data.method.config);
    heads = config.head_n;
    };

    fig = JSON.parse(data.fig);

    document.getElementById("result-card").style.display = 'block';
    var res = '<p>Изменения функции потерь:<\p>'
    heads.forEach((head, index) => {
    res += `<p>${head}: ${data.scores[index]}<\p>`
    });
    
    document.getElementById("result-card").innerHTML = res;

    document.getElementById("result-preview").style.display = 'block';
    Plotly.newPlot("result-preview", fig.data, fig.layout);

    get_history();
}

async function actmap(data = null) {
    if (data == null) {
    const input = document.getElementById('actmap-heads');
    const heads = input.value
    .split(',')
    .map(s => Number(s.trim()));

    data = await get_data(
        JSON.stringify({
        method: "actmap",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ head_n: heads, layer_n: parseInt(layer.value), prompt: document.getElementById('prompt-field').value })
        })
    );
    };

    console.log(data);

    document.getElementById("result-card").style.display = 'block';
    var res = `<p>Паттерны:<\p>
    <p>Текущий токен: ${(data.current.length!=0) ? data.current.toString() : 'Не найдены'}<\p>
    <p>Предыдущий токен: ${(data.prev.length!=0) ? data.prev.toString() : 'Не найдены'}<\p>
    <p>Первый токен: ${(data.first.length!=0) ? data.first.toString() : 'Не найдены'}<\p>
    <p>Индукция: ${(data.induction.length!=0) ? data.induction.toString() : 'Не найдены'}<\p>
    `;
    document.getElementById("result-card").innerHTML = res;

    document.getElementById("result-preview-iframe").style.display = 'block';
    document.getElementById("result-iframe").srcdoc = data.fig;

    get_history();
}

async function logattr(data = null) {
    if (data == null) {
    const input = document.getElementById('logattr-heads');
    const heads = input.value
    .split(',')
    .map(s => Number(s.trim()));

    data = await get_data(
        JSON.stringify({
        method: "logattr",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ head_n: heads, layer_n: parseInt(document.getElementById('logattr-layer').value), prompt: document.getElementById('prompt-field').value })
        })
    );
    };
    
    fig = JSON.parse(data.fig);

    document.getElementById("result-preview").style.display = 'block';
    Plotly.newPlot("result-preview", fig.data, fig.layout);

    get_history();
}

async function loglens(data = null) {
    if (data == null) {
    const input = document.getElementById('loglens-layer');

    data = await get_data(
        JSON.stringify({
        method: "loglens",
        cell_id: cellId,
        seq_no: 1,
        model_id: modelId,
        config: JSON.stringify({ layer_n: parseInt(input.value), prompt: document.getElementById('prompt-field').value })
        })
    );
    }
    
    fig = JSON.parse(data.fig);

    document.getElementById("result-preview").style.display = 'block';
    Plotly.newPlot("result-preview", fig.data, fig.layout);

    get_history();
}

async function actpatch(data = null) {
    if (data == null) {
    const input = document.getElementById('actpatch-heads');
    const heads = input.value
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
            layer_n: parseInt(document.getElementById('actpatch-layer').value), 
            correct_prompt: document.getElementById('actpatch-correct').value,
            corrupted_prompt: document.getElementById('actpatch-corrupted').value,
            answer: document.getElementById('actpatch-answer').value })
        })
    );
    }
    
    fig = JSON.parse(data.fig);

    document.getElementById("result-card").style.display = 'block';
    var res = `Среднее изменение: ${data.diff}`;
    document.getElementById("result-card").innerHTML = res;

    document.getElementById("result-preview").style.display = 'block';
    Plotly.newPlot("result-preview", fig.data, fig.layout);

    get_history();
}

async function get_history() {
    const resp = await fetch(`http://nginx:80/history`, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        cell_id: cellId
        })
    });

    data = await resp.json();
    
    const history_list = document.getElementById("history-list");
    history_list.innerHTML = '';
    document.getElementById("history-count").innerHTML=data.history.length;
    data.history.forEach(build_history);
}

function build_history(element) {
    const history_list = document.getElementById("history-list");
    const data = JSON.parse(element.config);

    const historyItem = document.createElement("div");
    historyItem.className = "history-item";
    historyItem.id = `history-item-${element.id}`;

    historyItem.addEventListener("click", () => {
    show(`history-item-${element.id}`);
    });

    const textDiv = document.createElement("div");
    textDiv.className = "text";
    textDiv.textContent = translate(element.method_name);

    const xDiv = document.createElement("div");
    xDiv.className = "x";
    xDiv.textContent = "×";

    historyItem.appendChild(textDiv);
    historyItem.appendChild(xDiv);

    const submenu = document.createElement("div");
    submenu.className = "submenu";
    submenu.id = `history-item-${element.id}-submenu`;

    for (const [key, value] of Object.entries(data)) {
    const p = document.createElement("p");
    p.textContent = `${translate(key)}: ${value}`;
    submenu.appendChild(p);
    }

    const replay_btn = document.createElement("button");
    replay_btn.textContent = "Запуск";
    replay_btn.addEventListener("click", () => {
    from_history(element.id);
    });
    submenu.appendChild(replay_btn);

    history_list.appendChild(historyItem);
    history_list.appendChild(submenu);
}

function translate(method_name) {
    switch (method_name) {
    case 'ablation': return 'Аблация';
    case 'actmap': return 'Карта активаций';
    case 'logattr': return 'Атрибуция логитов';
    case 'loglens': return 'Логитные линзы';
    case 'actpatch': return 'Патчинг активаций';
    
    case 'head_n': return 'Головы внимания';
    case 'layer_n': return 'Слои';
    case 'prompt': return 'Промпт';

    default: return method_name;
    }
}

async function from_history(id) {
    console.log(id);
    const resp = await fetch(`http://nginx:80/history/${id}`, {
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
    }
}