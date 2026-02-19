# 🎙️ Guia de Instalação — Qwen3-TTS Voice Clone

Guia passo a passo para instalar e rodar o Qwen3-TTS com interface web de clonagem de voz no Windows.

---

## ✅ Requisitos

| Item | Mínimo | Recomendado |
|------|--------|-------------|
| **SO** | Windows 10/11 | Windows 11 |
| **GPU** | NVIDIA com 6 GB VRAM | 8 GB+ VRAM |
| **RAM** | 16 GB | 32 GB |
| **Espaço em disco** | 10 GB livres | 20 GB livres |
| **CUDA** | 11.8 | 12.1+ |
| **Python** | 3.10 | 3.12 |

> ⚠️ **GPU NVIDIA é obrigatória.** O modelo não roda em CPU de forma prática (muito lento).

---

## 📦 Passo 1 — Instalar o Miniconda

Se você ainda não tem o Miniconda/Anaconda:

1. Baixe o instalador: https://www.anaconda.com/download  
2. Durante a instalação, marque **"Add to PATH"**
3. Após instalar, abra o **Anaconda Prompt** (ou PowerShell)

---

## 🐍 Passo 2 — Criar o Ambiente Python

Abra o **Anaconda Prompt** e execute:

```bash
conda create -n qwen3-tts-cuda python=3.12 -y
conda activate qwen3-tts-cuda
```

---

## 📥 Passo 3 — Clonar o Repositório

```bash
git clone https://github.com/SEU_USUARIO/Qwen3-TTS.git
cd Qwen3-TTS
```

> 💡 Se você recebeu o projeto como `.zip`, extraia e entre na pasta pelo terminal:
> ```
> cd C:\caminho\para\Qwen3-TTS
> ```

---

## ⚙️ Passo 4 — Instalar as Dependências

Com o ambiente ativado (`qwen3-tts-cuda`), dentro da pasta do projeto:

```bash
pip install -e .
```

Isso instala o pacote `qwen-tts` e todas as dependências necessárias.

---

## 🔥 Passo 5 — Instalar PyTorch com CUDA

**Verifique sua versão do CUDA** (abra o prompt e rode `nvcc --version` ou veja no painel da NVIDIA).

### Para CUDA 12.1+
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Para CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📂 Passo 6 — Baixar os Modelos

Os modelos serão baixados automaticamente na primeira execução. Mas se preferir baixar antes:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
```

> 📌 O modelo Base (~3.5 GB) é o único necessário para clonagem de voz.

---

## 🚀 Passo 7 — Ajustar e Rodar o Script

Abra o arquivo `start_voice_clone.ps1` num editor de texto e ajuste o caminho do ambiente conda para o seu usuário:

```powershell
# Linha 6 — troque "rapha" pelo seu nome de usuário Windows
$CondaEnvPath = "C:\Users\SEU_USUARIO\miniconda3\envs\qwen3-tts-cuda"
```

Depois, no PowerShell (dentro da pasta do projeto):

```powershell
.\start_voice_clone.ps1
```

> ⚠️ Se o PowerShell bloquear a execução, rode primeiro:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 🌐 Passo 8 — Acessar a Interface

Após o servidor iniciar, abra o navegador em:

```
http://127.0.0.1:7861
```

---

## 🎤 Como Usar — Voice Clone (Guia Rápido)

### Aba "Clone & Generate"
1. **Reference Audio** → Sobe um áudio da voz a clonar (5–15 segundos, WAV/MP3)
2. **Reference Text** → Digite exatamente o que foi dito no áudio
3. **Target Text** → O texto que você quer gerar com a voz clonada
4. **Language** → Selecione o idioma do texto alvo
5. **Instructions** → (Opcional) Ex: "Fale em tom solene e calmo"
6. Clique **Generate** e aguarde

### Aba "SRT Batch" — Geração em Lote por Legenda
1. **Reference Audio** → Mesma voz de referência
2. **SRT Subtitle File** → Seu arquivo `.srt` com as legendas
3. **Output Folder** → Pasta onde os áudios serão salvos (ex: `C:\Users\Você\Desktop\audios`)
4. **Output Format** → Escolha MP3, WAV ou MP4
5. Clique **Generate All** e acompanhe o progresso

### Aba "Save / Load Voice"
- **Save Voice File** → Salva a voz de referência como arquivo `.pt` para reusar depois
- **Load Voice & Generate** → Gera áudio usando um `.pt` salvo (mais rápido)

---

## 🔧 ffmpeg (Necessário para MP3/MP4)

Para salvar em MP3 ou MP4, o `ffmpeg` precisa estar instalado:

1. Baixe em: https://ffmpeg.org/download.html (versão Windows)
2. Extraia e adicione a pasta `bin/` ao PATH do sistema
3. Teste: abra o PowerShell e rode `ffmpeg -version`

Ou instale via winget:
```powershell
winget install ffmpeg
```

---

## 🆘 Problemas Comuns

| Erro | Solução |
|------|---------|
| `CUDA out of memory` | Feche outros programas pesados. Reinicie o servidor |
| `No module named 'qwen_tts'` | Verifique se o ambiente conda está ativado e rode `pip install -e .` novamente |
| `flash-attn not installed` | Normal no Windows. O aviso pode ser ignorado — o programa roda sem ela |
| Script não abre | Execute `Set-ExecutionPolicy RemoteSigned` no PowerShell como admin |
| Modelos não baixam | Use VPN ou baixe manualmente com `huggingface-cli download` |

---

## 💬 Idiomas Suportados

Chinês · Inglês · Japonês · Coreano · Alemão · Francês · Russo · Português · Espanhol · Italiano

---

*Baseado no [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) da Alibaba Qwen Team.*
