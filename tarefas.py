import json
import sqlite3
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "phi3"

# Conexão com o banco SQLite
conn = sqlite3.connect("tasks.db")
cursor = conn.cursor()

# Cria a tabela de tarefas (sem prioridade)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        status TEXT DEFAULT 'todo',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

SYSTEM_PROMPT = """Você é um classificador de mensagens para um gerenciador de tarefas.
Analise a mensagem do usuário e responda SOMENTE com um JSON válido, em um destes formatos:

1) Se o usuário quer SALVAR/ANOTAR/ADICIONAR uma tarefa:
{"action": "save", "task": "<apenas o trecho da mensagem que descreve a tarefa>"}

2) Se o usuário quer VER/LISTAR/MOSTRAR as tarefas salvas:
{"action": "list", "task": null}

Em "task", coloque somente a tarefa em si, sem expressões como "me lembra de", "anota aí", "adiciona", "preciso".

Exemplos:
Mensagem: "anota aí que eu preciso comprar leite e pão"
{"action": "save", "task": "comprar leite e pão"}

Mensagem: "me lembra de ligar para o dentista amanhã"
{"action": "save", "task": "ligar para o dentista amanhã"}

Mensagem: "quais são minhas tarefas?"
{"action": "list", "task": null}

Mensagem: "mostra a lista"
{"action": "list", "task": null}
"""


def interpret_message(user_input):
    """
    Pergunta ao Ollama se a mensagem é para salvar uma tarefa ou listar as tarefas.

    Returns:
        dict: {"action": "save" | "list", "task": str | None}
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "format": "json",   # força a saída em JSON
        "stream": False,    # resposta única, não em pedaços
        "options": {"temperature": 0},
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Falha ao conectar ao Ollama em {OLLAMA_URL}: {e}")

    if response.status_code != 200:
        raise Exception(f"Ollama retornou erro {response.status_code}: {response.text}")

    content = response.json().get("message", {}).get("content", "")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Resposta do modelo não é um JSON válido: {content!r}")

    action = str(data.get("action", "")).lower().strip()
    task = data.get("task")
    task = task.strip() if isinstance(task, str) else None

    if action not in ("save", "list"):
        raise ValueError(f"Ação desconhecida retornada pelo modelo: {content!r}")
    if action == "save" and not task:
        raise ValueError(f"O modelo não identificou nenhuma tarefa: {content!r}")

    return {"action": action, "task": task}


def save_task(title):
    """Salva uma tarefa no banco e retorna o id."""
    cursor.execute("INSERT INTO tasks (title) VALUES (?)", (title,))
    conn.commit()
    return cursor.lastrowid


def list_tasks():
    """Retorna todas as tarefas salvas."""
    cursor.execute("SELECT id, title, status, created_at FROM tasks ORDER BY id")
    return cursor.fetchall()


def handle_message(message):
    result = interpret_message(message)

    if result["action"] == "save":
        task_id = save_task(result["task"])
        print(f"Tarefa salva (#{task_id}): {result['task']}")
    else:
        tasks = list_tasks()
        if not tasks:
            print("Nenhuma tarefa salva ainda.")
            return
        print("Suas tarefas:")
        for task_id, title, status, created_at in tasks:
            print(f"  #{task_id} [{status}] {title}  ({created_at})")


# Loop principal
print("Bem-vindo! Digite 'exit' ou 'quit' (ou Ctrl+C) para sair.")
try:
    while True:
        message = input("> ").strip()
        if not message:
            continue
        if message.lower() in ("exit", "quit"):
            print("Até logo!")
            break

        try:
            handle_message(message)
        except (ValueError, ConnectionError, Exception) as e:
            print(f"Erro: {e}")
except KeyboardInterrupt:
    print("\nSaindo...")
finally:
    conn.close()