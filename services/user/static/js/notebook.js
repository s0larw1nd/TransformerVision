document.getElementById("btn-back").addEventListener("click", () => back());
document.getElementById("btn-insert-text-first").addEventListener("click", () => insert_text_first());
document.getElementById("btn-insert-model-first").addEventListener("click", () => insert_model_first());

document.getElementsByName("btn-insert-text").forEach(button => {
    button.addEventListener(
        "click",
        () => {
            let id = button.dataset.cellId;
            insert_text_after(id);
        }
    );
});
document.getElementsByName("btn-insert-model").forEach(button => {
    button.addEventListener(
        "click",
        () => {
            let id = button.dataset.cellId;
            insert_model_after(id);
        }
    );
});

const maxId = data.length ? Math.max(...data.map(item => item.id)) : 0;
var inc = 1;

function back() {
    history.back();
}

function insert_text_first() {
    const current = document.getElementById(`insert-zone-first`);
    const newId = maxId + inc;
    inc += 1;

    const html = `
    <article class="cell" id="cell-${newId}" data-type="TEXT">
    <div class="cell-header">
        <span class="cell-type">
        <span class="marker"></span> Текстовая ячейка
        </span>
        <div class="cell-actions">
        <button class="btn cell-btn" type="button">Скрыть</button>
        <button class="btn cell-btn danger" type="button">Удалить</button>
        </div>
    </div>
    <div class="cell-content" id="cell-content-${newId}">
        Текст
    </div>
    </article>

    <div class="insert-zone" id="insert-zone-${newId}">
    <div class="insert-actions">
        <button class="btn insert-small" type="button" onclick="insert_text_after( ${newId} )">+ Текстовая ячейка</button>
        <button class="btn insert-small" type="button" onclick="insert_model_after( ${newId} )">+ Модельная ячейка</button>
    </div>
    </div>
    `;

    current.insertAdjacentHTML('afterend', html);
    add_cell_listeners(newId);
    syncCells();
}
function insert_model_first() {
    const current = document.getElementById(`insert-zone-first`);
    const newId = maxId + inc;
    inc += 1;

    const html = `
    <article class="cell" id="cell-${newId}" data-type="MODEL">
    <div class="cell-header">
        <span class="cell-type"><span class="marker"></span> Модельная ячейка</span>
        <div class="cell-actions">
        <button class="btn cell-btn" type="button">Скрыть</button>
        <button class="btn cell-btn danger" type="button">Удалить</button>
        </div>
    </div>

    <div class="model-cell">
        <a class="model-link" href="/project/${projectId}/experiment/${experimentId}/cell/${newId}/model">Открыть UI работы с моделью</a>
        <div class="empty-note">Метрики не добавлены</div>
    </div>
    </article>

    <div class="insert-zone" id="insert-zone-${newId}">
    <div class="insert-actions">
        <button class="btn insert-small" type="button" onclick="insert_text_after( ${newId} )">+ Текстовая ячейка</button>
        <button class="btn insert-small" type="button" onclick="insert_model_after( ${newId} )">+ Модельная ячейка</button>
    </div>
    </div>
    `;

    current.insertAdjacentHTML('afterend', html);
    syncCells();
}

function insert_text_after(after) {
    const current = document.getElementById(`insert-zone-${after}`);
    const newId = maxId + inc;
    inc += 1;

    const html = `
    <article class="cell" id="cell-${newId}" data-type="TEXT">
    <div class="cell-header">
        <span class="cell-type">
        <span class="marker"></span> Текстовая ячейка
        </span>
        <div class="cell-actions">
        <button class="btn cell-btn" type="button">Скрыть</button>
        <button class="btn cell-btn danger" type="button">Удалить</button>
        </div>
    </div>
    <div class="cell-content" id="cell-content-${newId}">
        Текст
    </div>
    </article>

    <div class="insert-zone" id="insert-zone-${newId}">
    <div class="insert-actions">
        <button class="btn insert-small" type="button" onclick="insert_text_after( ${newId} )">+ Текстовая ячейка</button>
        <button class="btn insert-small" type="button" onclick="insert_model_after( ${newId} )">+ Модельная ячейка</button>
    </div>
    </div>
    `;
    
    current.insertAdjacentHTML('afterend', html);
    add_cell_listeners(newId);
    syncCells();
}
function insert_model_after(after) {
    const current = document.getElementById(`insert-zone-${after}`);
    const newId = maxId + inc;
    inc += 1;

    const html = `
    <article class="cell" id="cell-${newId}" data-type="MODEL">
    <div class="cell-header">
        <span class="cell-type"><span class="marker"></span> Модельная ячейка</span>
        <div class="cell-actions">
        <button class="btn cell-btn" type="button">Скрыть</button>
        <button class="btn cell-btn danger" type="button">Удалить</button>
        </div>
    </div>

    <div class="model-cell">
        <a class="model-link" href="/project/${projectId}/experiment/${experimentId}/cell/${newId}/model">Открыть UI работы с моделью</a>
        <div class="empty-note">Метрики не добавлены</div>
    </div>
    </article>

    <div class="insert-zone" id="insert-zone-${newId}">
    <div class="insert-actions">
        <button class="btn insert-small" type="button" onclick="insert_text_after( ${newId} )">+ Текстовая ячейка</button>
        <button class="btn insert-small" type="button" onclick="insert_model_after( ${newId} )">+ Модельная ячейка</button>
    </div>
    </div>
    `;

    current.insertAdjacentHTML('afterend', html);
    syncCells();
}

function collectCells() {
    const articles = document.querySelectorAll('article.cell');

    const result = [];

    articles.forEach((article, index) => {
    let type;
    let stext = null;

    type = article.dataset.type;

    if (type === "TEXT") {
        const content = article.querySelector('.cell-content');
        stext = content ? content.innerText.trim() : "";
    }

    result.push({
        id: parseInt(article.id.replace('cell-', '')),
        pos: index,
        type: type,
        stext: stext
    });
    });

    console.log(result);

    return result;
}

function add_cell_listeners(id) {
    console.log(id);
    const cell = document.getElementById(`cell-${id}`);
    const content = document.getElementById(`cell-content-${id}`);

    cell.addEventListener('dblclick', () => {
    content.contentEditable = true;
    cell.classList.add('editing');

    content.focus();

    const range = document.createRange();
    range.selectNodeContents(content);
    range.collapse(false);

    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    });

    cell.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        content.contentEditable = false;
        cell.classList.remove('editing');
        cell.blur();
        syncCells();
    }
    });

    cell.addEventListener('blur', () => {
    cell.contentEditable = false;
    cell.classList.remove('editing');
    syncCells();
    });
}

async function syncCells() {
    const data = collectCells();

    await fetch(`/project/${projectId}/experiment/${experimentId}/sync`, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
    });
}