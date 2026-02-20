# DL_Proyecto

Asistente RAG para responder preguntas sobre **el primer libro de Juego de Tronos** desde un `.epub` y generar una imagen de la escena más representativa.

## Modelos configurados

- LLM: `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- Imagen: `stabilityai/stable-diffusion-3.5-large`

Ambos se consumen por defecto vía **Hugging Face Inference API** (no requiere cargar localmente esos modelos gigantes).

## Qué mejora este enfoque

- Recupera automáticamente contexto relevante del libro para cualquier pregunta.
- Separa respuesta textual y prompt visual.
- Genera un **prompt de imagen genérico** (no hardcodeado a escenas concretas), construido por el LLM con el contexto recuperado.
- Mantiene trazabilidad de fragmentos recuperados y guarda resultado en JSON.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido (script)

```bash
python got_assistant.py \
  --epub /ruta/a/tu/libro1.epub \
  --question "¿Cómo se presenta a Tyrion en su primera aparición?" \
  --hf-token "hf_xxx"
```

Salida:
- `outputs/result.json` con respuesta + evidencias + prompt de imagen.
- `outputs/scene.png` con la escena generada.

## Notebook

Archivo: `demo_got_assistant.ipynb`

Ajusta:
- `HF_TOKEN`
- `EPUB_PATH`
- `QUESTION`

Y ejecuta celdas en orden.

## Notas

- Si quieres cambiar modelos, usa flags:
  - `--qwen-model`
  - `--image-model`
  - `--embedding-model`
- Si no tienes créditos/permiso para inferencia remota en Hugging Face, la llamada fallará aunque el código sea correcto.
