document.getElementById("btn-cancel").addEventListener("click", () => back());
document.getElementById("btn-back").addEventListener("click", () => back());

function back() {
    history.back();
}