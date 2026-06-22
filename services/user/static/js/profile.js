document.getElementById("btn-logout").addEventListener("click", () => logout());
document.getElementById("empty-card").addEventListener("click", () => create_project());

document.getElementsByName("article").forEach(article => {
    article.addEventListener(
        "click",
        () => {
            let id = article.dataset.projectId;
            open_project(id);
        }
    );
});

function open_project(proj_id) {
    window.location.href = `/project/${proj_id}`;
}

function create_project() {
    window.location.href = "/project/create";
}

function logout() {
    window.location.href = "/logout";
}