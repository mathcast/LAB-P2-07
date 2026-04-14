# LAB P2-07: Fine-Tuning de LLM com LoRA e QLoRA

## Como baixar o projeto

```bash
git clone https://github.com/mathcast/LAB-P2-07.git
cd LAB-P2-07
```

## Descrição do projeto

Este projeto implementa o processo completo de fine-tuning de um modelo de linguagem (LLM) utilizando:

- LoRA (Low-Rank Adaptation)
- QLoRA (quantização em 4 bits)

O objetivo é treinar um modelo baseado em Llama 2 7B Chat para responder perguntas sobre APIs, utilizando um dataset gerado automaticamente via API.

## Conceitos utilizados

### LoRA (Low-Rank Adaptation)

LoRA reduz o custo de treinamento ao ajustar apenas matrizes de baixa dimensão, mantendo os pesos originais congelados.

#### Parâmetros utilizados:

```bash
r = 64
alpha = 16
dropout = 0.1
```
### QLoRA (Quantized LoRA)

#### QLoRA permite treinar modelos grandes usando menos memória ao aplicar:

- Quantização em 4 bits
- Tipo de quantização: nf4
- Computação em float16


## Estrutura do repositório

```bash
LAB P2-07/
├── scripts/
│   ├── generate_dataset.py   # Geração do dataset usando API (Groq)
│   ├── train.py              # Treinamento com LoRA + QLoRA
│   └── test_model.py         # Teste do modelo treinado
│
├── data/
│   ├── train.jsonl           # Dataset de treino
│   └── test.jsonl            # Dataset de teste
│
├── lora-model/               # Modelo treinado (gerado após treino)
│
├── requirements.txt          # Dependências
└── README.md
```

## Formato do dataset

Os dados seguem o formato .jsonl, onde cada linha é um JSON:

```bash
{"prompt": "O que é uma API?", "response": "Uma API é um conjunto de regras que permite a comunicação entre sistemas."}
```

Durante o treinamento, os dados são transformados para:

```bash
Pergunta: O que é uma API?
Resposta: Uma API é...
``` 
## Como rodar
1. Criar ambiente virtual

Windows (PowerShell):

```bash
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

2. Instalar dependências

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.2 datasets==2.16.1 peft==0.7.1 trl==0.7.10 accelerate==0.25.0 bitsandbytes groq
```

3. Gerar dataset

```bash
python scripts/generate_dataset.py
```

Isso irá criar:

```bash
data/train.jsonl
data/test.jsonl
```

4. Treinar o modelo

```bash
python scripts/train.py
```

Durante o treino, será exibido:

- progresso (steps)
- loss do modelo

Ao final:

```bash
Treinamento finalizado!
```

E será criada a pasta:

```bash
lora-model/
```

5. Testar o modelo

```bash
python scripts/test_model.py
```
Exemplo de saída:

```bash
Pergunta: O que é uma API?
Resposta: Uma API é um conjunto de regras que permite...
```
## Requisitos técnicos
- Linguagem: Python 3.10
- GPU: recomendada (ex: RTX 4060)
- Frameworks:
    - PyTorch
    - Transformers
    - PEFT (LoRA)
    - TRL
    - Datasets

## Observações

O treinamento utiliza QLoRA, permitindo rodar modelos grandes com menor uso de memória.
O dataset é gerado automaticamente utilizando API externa (Groq).
O modelo base utilizado foi o Llama 2 7B Chat.

## Uso de IA

Partes do código foram geradas com auxílio de Inteligência Artificial e posteriormente revisadas e ajustadas manualmente.
