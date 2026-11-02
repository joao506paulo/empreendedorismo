#!/usr/bin/env python3
"""
Instala e configura o Ollama para uso com o task_manager.py.

O que o script faz:
  1. Verifica se o Ollama está instalado (e instala, se não estiver)
  2. Garante que o servidor local do Ollama está rodando
  3. Baixa o modelo escolhido (padrão: phi3)
  4. Instala a dependência Python 'requests'
  5. Faz um teste rápido para confirmar que tudo funciona

Uso:
  python setup_ollama.py
  python setup_ollama.py --model llama3.1
  python setup_ollama.py --skip-test
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA_HOST = "http://localhost:11434"


def log(msg):
    print(f"[setup] {msg}", flush=True)


def fail(msg):
    print(f"[erro] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def find_ollama():
    """Retorna o caminho do executável do Ollama, ou None se não for encontrado."""
    path = shutil.which("ollama")
    if path:
        return path

    # Em alguns sistemas o PATH só é atualizado depois de reabrir o terminal
    candidates = []
    system = platform.system()
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA", "")
        candidates.append(os.path.join(local, "Programs", "Ollama", "ollama.exe"))
    elif system == "Darwin":
        candidates += ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama"]
        candidates.append("/Applications/Ollama.app/Contents/Resources/ollama")
    else:
        candidates += ["/usr/local/bin/ollama", "/usr/bin/ollama"]

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def install_ollama():
    system = platform.system()
    log(f"Ollama não encontrado. Instalando para {system}...")

    if system == "Linux":
        if not shutil.which("curl"):
            fail("O 'curl' é necessário para instalar o Ollama. Instale-o e rode o script de novo.")
        # Script oficial de instalação (pode pedir sua senha de sudo)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

    elif system == "Darwin":
        if shutil.which("brew"):
            subprocess.run(["brew", "install", "ollama"], check=True)
        else:
            fail(
                "Homebrew não encontrado. Baixe o instalador em https://ollama.com/download "
                "e rode este script novamente."
            )

    elif system == "Windows":
        if shutil.which("winget"):
            subprocess.run(
                [
                    "winget", "install", "-e", "--id", "Ollama.Ollama",
                    "--accept-package-agreements", "--accept-source-agreements",
                ],
                check=True,
            )
        else:
            fail(
                "winget não encontrado. Baixe o instalador em https://ollama.com/download "
                "e rode este script novamente."
            )
    else:
        fail(f"Sistema não suportado: {system}. Veja https://ollama.com/download")

    exe = find_ollama()
    if not exe:
        fail(
            "A instalação terminou, mas o executável ainda não foi encontrado. "
            "Feche e reabra o terminal e rode este script novamente."
        )
    return exe


def server_is_up():
    try:
        with urllib.request.urlopen(OLLAMA_HOST, timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def ensure_server(ollama_exe):
    if server_is_up():
        log("Servidor do Ollama já está rodando.")
        return

    log("Iniciando o servidor do Ollama em segundo plano...")
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([ollama_exe, "serve"], **kwargs)

    for _ in range(30):
        if server_is_up():
            log("Servidor iniciado.")
            return
        time.sleep(1)
    fail("O servidor do Ollama não respondeu em 30 segundos.")


def model_is_present(ollama_exe, model):
    result = subprocess.run([ollama_exe, "list"], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    wanted = model if ":" in model else f"{model}:latest"
    for line in result.stdout.splitlines()[1:]:
        name = line.split()[0] if line.split() else ""
        if name == wanted or name == model:
            return True
    return False


def pull_model(ollama_exe, model):
    if model_is_present(ollama_exe, model):
        log(f"Modelo '{model}' já está baixado.")
        return
    log(f"Baixando o modelo '{model}' (pode demorar alguns minutos)...")
    subprocess.run([ollama_exe, "pull", model], check=True)


def install_python_deps():
    try:
        import requests  # noqa: F401
        log("Dependência 'requests' já instalada.")
    except ImportError:
        log("Instalando a dependência 'requests'...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)


def test_model(model):
    log("Testando o modelo com uma mensagem de exemplo...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Responda apenas com a palavra: ok"}],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        answer = data.get("message", {}).get("content", "").strip()
        log(f"Resposta do modelo: {answer!r}")
    except Exception as e:
        fail(f"O teste falhou: {e}")


def main():
    parser = argparse.ArgumentParser(description="Instala e configura o Ollama para o task_manager.py")
    parser.add_argument("--model", default="phi3", help="modelo a baixar (padrão: phi3)")
    parser.add_argument("--skip-test", action="store_true", help="pula o teste final")
    args = parser.parse_args()

    ollama_exe = find_ollama()
    if ollama_exe:
        log(f"Ollama encontrado em: {ollama_exe}")
    else:
        try:
            ollama_exe = install_ollama()
        except subprocess.CalledProcessError as e:
            fail(f"A instalação falhou (código {e.returncode}).")

    ensure_server(ollama_exe)

    try:
        pull_model(ollama_exe, args.model)
    except subprocess.CalledProcessError as e:
        fail(f"Não foi possível baixar o modelo '{args.model}' (código {e.returncode}).")

    install_python_deps()

    if not args.skip_test:
        test_model(args.model)

    log("Tudo pronto! Agora rode: python task_manager.py")
    if args.model != "phi3":
        log(f"Lembre de trocar MODEL = \"{args.model}\" no task_manager.py.")


if __name__ == "__main__":
    main()
