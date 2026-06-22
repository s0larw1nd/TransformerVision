document.getElementById("btn-back").addEventListener("click", () => back());
document.getElementById("btn-create-experiment").addEventListener("click", () => create_experiment());

document.getElementsByName("article").forEach(article => {
    article.addEventListener(
        "click",
        () => {
            let id = article.dataset.experimentId;
            open_experiment(id);
        }
    );
});

function back() {
    history.back();
}

function open_experiment(exp_id) {
    window.location.href = `/project/${projectId}/experiment/${exp_id}`;
}

async function create_experiment() {
    const resp = await fetch(`/project/${projectId}/experiment/create`, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({ })
    });

    window.location.reload();
}