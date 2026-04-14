from groq import Groq
import json
import random
import os

# pega a API key da variável de ambiente (ou coloca direto se quiser)
client = Groq(api_key="")

dataset = []

for i in range(60):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Responda apenas com JSON válido, sem explicação."
            },
            {
                "role": "user",
                "content": "Gere uma pergunta e resposta sobre APIs no formato: {\"prompt\": \"...\", \"response\": \"...\"}"
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    try:
        # tenta extrair JSON mesmo se vier com texto extra
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]

        data = json.loads(json_str)
        dataset.append(data)

    except:
        print("Erro ao processar:", content)
        continue

# mostra quantos dados válidos foram gerados
print(f"Total de exemplos válidos: {len(dataset)}")

# embaralha
random.shuffle(dataset)

# divide 90% treino / 10% teste
split = int(len(dataset) * 0.9)

train = dataset[:split]
test = dataset[split:]

# garante que a pasta existe
os.makedirs("data", exist_ok=True)

# função para salvar jsonl
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# salva arquivos
save_jsonl(train, "data/train.jsonl")
save_jsonl(test, "data/test.jsonl")

print("Dataset gerado com sucesso!")