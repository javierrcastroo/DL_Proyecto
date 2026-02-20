"""RAG assistant for *A Game of Thrones* with answer + image generation.

Supports Hugging Face hosted inference by default for:
- Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
- stabilityai/stable-diffusion-3.5-large

It reads an EPUB, builds a semantic index, retrieves context for each user
question, answers grounded on the retrieved passages, and creates a generic,
context-aware image prompt for the same scene.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import ebooklib
import faiss
import numpy as np
from bs4 import BeautifulSoup
from ebooklib import epub
from huggingface_hub import InferenceClient
from PIL import Image
from sentence_transformers import SentenceTransformer


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
DEFAULT_IMAGE_MODEL = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@dataclass
class RetrievedChunk:
    text: str
    score: float
    index: int


def clean_text(raw: str) -> str:
    """Normalize whitespace and remove noisy artifacts."""
    text = re.sub(r"\s+", " ", raw)
    text = re.sub(r"\u00a0", " ", text)
    return text.strip()


def extract_text_from_epub(epub_path: Path) -> str:
    """Extract plain text from all XHTML docs inside EPUB."""
    book = epub.read_epub(str(epub_path))
    all_parts: list[str] = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html = item.get_content()
            soup = BeautifulSoup(html, "html.parser")
            part = soup.get_text(separator=" ")
            part = clean_text(part)
            if part:
                all_parts.append(part)

    text = "\n\n".join(all_parts)
    if not text:
        raise ValueError(f"No se pudo extraer texto del EPUB: {epub_path}")
    return text


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks suitable for retrieval."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size debe ser mayor que overlap")

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunk = clean_text(chunk)
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap

    return chunks


class BookRetriever:
    """Semantic retriever for book chunks."""

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[str] = []

    def build(self, chunks: Iterable[str]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("No hay chunks para indexar")

        embeddings = self.embedder.encode(
            self.chunks,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError("El índice no está construido")

        q_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, ids = self.index.search(q_emb, top_k)
        result: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0 or idx >= len(self.chunks):
                continue
            result.append(RetrievedChunk(text=self.chunks[idx], score=float(score), index=int(idx)))
        return result


class GoTAssistant:
    """Question answering + image generation assistant."""

    def __init__(
        self,
        hf_token: str | None,
        qwen_model: str = DEFAULT_QWEN_MODEL,
        image_model: str = DEFAULT_IMAGE_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.qwen_model = qwen_model
        self.image_model = image_model
        self.client = InferenceClient(token=hf_token)
        self.retriever = BookRetriever(embedding_model=embedding_model)

    def load_book(self, epub_path: Path, chunk_size: int = 1200, overlap: int = 150) -> None:
        text = extract_text_from_epub(epub_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        self.retriever.build(chunks)

    def build_answer_prompt(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        context_blocks = "\n\n".join(
            f"[Fragmento {i+1} | score={c.score:.3f}]\n{c.text}"
            for i, c in enumerate(retrieved)
        )
        return f"""
Eres un asistente experto en *A Song of Ice and Fire*.
Responde SIEMPRE en español y solamente con información respaldada por el contexto.
Si no hay datos suficientes, dilo claramente.

Pregunta del usuario:
{question}

Contexto recuperado del libro:
{context_blocks}

Instrucciones de salida:
1) Responde la pregunta con precisión y brevedad.
2) Incluye detalles de personajes, lugar y atmósfera cuando el contexto lo permita.
3) Añade una sección final llamada "Evidencia" con bullets cortos citando qué fragmentos apoyan la respuesta.
""".strip()

    def build_generic_image_prompt_request(
        self,
        question: str,
        answer: str,
        retrieved: list[RetrievedChunk],
    ) -> str:
        """Generic prompt builder for any question.

        The LLM transforms RAG context + answer into a cinematic text-to-image prompt.
        """
        context_blocks = "\n\n".join(
            f"[Fragmento {i+1}]\n{c.text}"
            for i, c in enumerate(retrieved)
        )
        return f"""
Tu tarea es crear un prompt de imagen *genérico y reutilizable* para un modelo diffusion.
Debes capturar la escena más representativa para ilustrar la pregunta del usuario,
utilizando la respuesta y el contexto del libro.

Pregunta del usuario:
{question}

Respuesta generada:
{answer}

Contexto relevante:
{context_blocks}

Devuelve un JSON válido con estas claves exactas:
- "positive_prompt": descripción visual rica, cinematográfica, concreta y fiel al contexto.
- "negative_prompt": elementos a evitar (artefactos, anacronismos, estilo no deseado).
- "style": breve etiqueta de estilo visual.
- "shot": tipo de plano/cámara.

Reglas:
- Evita spoilers fuera del contexto proporcionado.
- Evita contradicciones con los fragmentos.
- Si faltan detalles, inventa lo mínimo manteniendo coherencia medieval-fantasía.
- No incluyas markdown, solo JSON.
""".strip()

    def ask(self, question: str, top_k: int = 5, max_new_tokens: int = 600) -> dict:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved:
            raise RuntimeError("No se recuperó contexto del libro")

        answer_prompt = self.build_answer_prompt(question, retrieved)
        answer = self.client.text_generation(
            prompt=answer_prompt,
            model=self.qwen_model,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
        )

        image_prompt_request = self.build_generic_image_prompt_request(question, answer, retrieved)
        image_prompt_json = self.client.text_generation(
            prompt=image_prompt_request,
            model=self.qwen_model,
            max_new_tokens=300,
            temperature=0.3,
        )

        parsed = self._safe_parse_prompt_json(image_prompt_json)
        return {
            "question": question,
            "answer": answer.strip(),
            "retrieved": [c.__dict__ for c in retrieved],
            "image_prompt": parsed,
        }

    def generate_image(
        self,
        positive_prompt: str,
        negative_prompt: str,
        output_path: Path,
        guidance_scale: float = 5.5,
        num_inference_steps: int = 40,
        width: int = 1024,
        height: int = 1024,
    ) -> Path:
        image: Image.Image = self.client.text_to_image(
            prompt=positive_prompt,
            model=self.image_model,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path

    @staticmethod
    def _safe_parse_prompt_json(raw: str) -> dict:
        text = raw.strip()
        # Remove accidental code fences.
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {
                "positive_prompt": text,
                "negative_prompt": "blurry, low quality, watermark, text artifacts, modern objects",
                "style": "cinematic fantasy realism",
                "shot": "medium shot",
            }

        required = {"positive_prompt", "negative_prompt", "style", "shot"}
        missing = required - set(data.keys())
        for key in missing:
            data[key] = ""
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GoT RAG + image generation assistant")
    parser.add_argument("--epub", type=Path, required=True, help="Ruta al EPUB del libro")
    parser.add_argument("--question", type=str, required=True, help="Pregunta del usuario")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qwen-model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--image-model", type=str, default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/result.json"))
    parser.add_argument("--output-image", type=Path, default=Path("outputs/scene.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assistant = GoTAssistant(
        hf_token=args.hf_token,
        qwen_model=args.qwen_model,
        image_model=args.image_model,
        embedding_model=args.embedding_model,
    )
    assistant.load_book(args.epub)

    result = assistant.ask(args.question, top_k=args.top_k)
    assistant.generate_image(
        positive_prompt=result["image_prompt"]["positive_prompt"],
        negative_prompt=result["image_prompt"]["negative_prompt"],
        output_path=args.output_image,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Respuesta:\n")
    print(result["answer"])
    print("\nPrompt de imagen:\n")
    print(json.dumps(result["image_prompt"], ensure_ascii=False, indent=2))
    print(f"\nImagen guardada en: {args.output_image}")
    print(f"Resultado JSON guardado en: {args.output_json}")


if __name__ == "__main__":
    main()
