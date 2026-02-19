from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# キャラクター（システムプロンプト）
# ----------------------------
PERSONAS: dict[str, str] = {
    "default": "あなたは丁寧で親切なアシスタントです。簡潔に、分かりやすく答えてください。",
    "cat": "あなたは猫のキャラクターのチャットボットです。語尾に「にゃ」を付けて、親しみやすく可愛く話してください。",
}

# ----------------------------
# ユーティリティ
# ----------------------------
def get_openai_api_key() -> str:
    key = os.environ.get("API_KEY")
    if not key:
        raise RuntimeError("環境変数 API_KEY が設定されていません。")
    return key


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


# ----------------------------
# Web検索（Tavily）
# ----------------------------
@dataclass
class WebResult:
    title: str
    url: str
    content: str


def tavily_search(query: str, api_key: str, max_results: int = 3) -> list[WebResult]:
    """
    Tavily Search APIを直接叩く最小実装。
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()

    results: list[WebResult] = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            WebResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", "") or item.get("snippet", "") or "",
            )
        )
    return results


# ----------------------------
# RAG（ローカル文書の簡易ベクトル検索）
# ----------------------------
@dataclass
class Chunk:
    source: str  # 例: ファイル名
    text: str
    embedding: list[float]


class RagIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        embed_texts: Callable[[list[str]], list[list[float]]],
        exts: set[str] = {".txt", ".md"},
        chunk_size: int = 800,
        overlap: int = 120,
        embed_batch_size: int = 64,
    ) -> "RagIndex":
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return cls([])

        texts: list[str] = []
        sources: list[str] = []

        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                raw = p.read_text(encoding="utf-8", errors="ignore")
                for c in split_text(raw, chunk_size=chunk_size, overlap=overlap):
                    texts.append(c)
                    # 相対パスで持っておくと見やすい
                    sources.append(str(p.relative_to(data_dir)))

        if not texts:
            return cls([])

        embeddings: list[list[float]] = []
        for batch in batched(texts, embed_batch_size):
            embeddings.extend(embed_texts(batch))

        chunks = [
            Chunk(source=sources[i], text=texts[i], embedding=embeddings[i])
            for i in range(len(texts))
        ]
        return cls(chunks)

    def search(
        self,
        query: str,
        embed_texts: Callable[[list[str]], list[list[float]]],
        top_k: int = 4,
    ) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q_emb = embed_texts([query])[0]
        scored = [(c, cosine_similarity(q_emb, c.embedding)) for c in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ----------------------------
# OpenAI Backend（Chat + Embeddings）
# ----------------------------
class OpenAIBackend:
    def __init__(self, api_key: str, chat_model: str, embedding_model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        res = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
        )
        return (res.choices[0].message.content or "").strip()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [d.embedding for d in res.data]


# ----------------------------
# チャットボット本体（RAG + Web検索）
# ----------------------------
class RagWebChatBot:
    def __init__(
        self,
        backend: OpenAIBackend,
        data_dir: str | Path = "./data/text",
        persona: str = "default",
        max_turns: int = 8,
        web_mode: str = "auto",  # "off" | "auto" | "on"
        tavily_api_key: str | None = None,
        index: RagIndex | None = None,  # テスト用に注入可
    ) -> None:
        self.backend = backend
        self.max_turns = max_turns
        self.web_mode = web_mode
        self.tavily_api_key = tavily_api_key

        if persona not in PERSONAS:
            persona = "default"
        self.persona_name = persona
        self.system_prompt = PERSONAS[persona]

        # 履歴（systemは固定、historyには入れない）
        self.history: list[dict[str, str]] = []

        # RAG index
        self.index = index or RagIndex.from_directory(data_dir, embed_texts=self.backend.embed_texts)

    def _trim_history(self) -> None:
        max_msgs = self.max_turns * 2
        while len(self.history) > max_msgs:
            self.history.pop(0)

    def _should_web_search(self, user_text: str) -> bool:
        keywords = ["検索", "調べ", "Web", "ウェブ", "最新", "今日", "今", "ニュース", "価格", "相場", "天気"]
        return any(k in user_text for k in keywords)

    def _context_message(self, rag_hits: list[tuple[Chunk, float]], web_hits: list[WebResult]) -> str:
        parts: list[str] = []
        parts.append("以下は参考情報です。必要に応じて参照して回答してください。")

        if rag_hits:
            parts.append("\n【RAG（ローカル文書）】")
            for i, (chunk, score) in enumerate(rag_hits, 1):
                snippet = chunk.text[:700].replace("\n", " ")
                parts.append(f"[DOC{i}] source={chunk.source} score={score:.3f}\n{snippet}")
        else:
            parts.append("\n【RAG（ローカル文書）】該当なし")

        if web_hits:
            parts.append("\n【Web検索】")
            for i, r in enumerate(web_hits, 1):
                snippet = (r.content or "")[:700].replace("\n", " ")
                parts.append(f"[WEB{i}] {r.title}\nURL={r.url}\n{snippet}")
        else:
            parts.append("\n【Web検索】未実行または結果なし")

        parts.append("根拠が不足する場合は、その旨を明確に述べてください。")
        return "\n".join(parts)

    def handle_command(self, text: str) -> str | None:
        if not text.startswith("/"):
            return None

        cmd, *rest = text.strip().split(maxsplit=1)
        arg = rest[0] if rest else ""

        if cmd == "/help":
            return (
                "コマンド:\n"
                "  /help                ヘルプ\n"
                "  /reset               会話履歴をリセット\n"
                "  /persona <default|cat> キャラ変更\n"
                "  /web <on|off|auto>   Web検索モード\n"
                "  /exit                終了\n"
                "※通常は文章を入力すると回答します。"
            )

        if cmd == "/reset":
            self.history = []
            return "会話履歴をリセットしました。"

        if cmd == "/persona":
            if arg not in PERSONAS:
                return f"未定義のキャラです（{list(PERSONAS.keys())}）"
            self.persona_name = arg
            self.system_prompt = PERSONAS[arg]
            self.history = []
            return f"キャラクターを {arg} に変更しました（履歴もリセット）。"

        if cmd == "/web":
            if arg not in {"on", "off", "auto"}:
                return "使い方: /web on | off | auto"
            self.web_mode = arg
            return f"Web検索モード: {arg}"

        if cmd == "/exit":
            return "__EXIT__"

        return f"不明なコマンドです: {cmd}（/help を参照）"

    def reply(self, user_text: str) -> str:
        # コマンド
        cmd = self.handle_command(user_text)
        if cmd is not None:
            return cmd

        user_text = user_text.strip()
        if not user_text:
            return ""

        # 1) RAG検索（常に実行）
        rag_hits = self.index.search(user_text, embed_texts=self.backend.embed_texts, top_k=4)

        # 2) Web検索（モードに応じて）
        web_hits: list[WebResult] = []
        can_web = bool(self.tavily_api_key)
        need_web = (
            self.web_mode == "on"
            or (self.web_mode == "auto" and self._should_web_search(user_text))
        )
        if can_web and need_web:
            try:
                web_hits = tavily_search(user_text, api_key=self.tavily_api_key, max_results=3)
            except Exception:
                web_hits = []

        # 3) LLMへ投げるメッセージ構築
        messages_for_api: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            *self.history,
            {"role": "system", "content": self._context_message(rag_hits, web_hits)},
            {"role": "user", "content": user_text},
        ]

        try:
            answer = self.backend.chat(messages_for_api)
        except Exception as e:
            answer = f"エラー: API呼び出しに失敗しました（{type(e).__name__}）"

        # 履歴更新
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return answer

    def save_transcript(self, out_dir: str | Path = ".") -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / datetime.now().strftime("chat_%Y%m%d_%H%M%S.jsonl")

        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "system", "content": self.system_prompt}, ensure_ascii=False) + "\n")
            for m in self.history:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        return path


def main() -> None:
    load_dotenv("../.env")  # 無ければ無視されます
    api_key = get_openai_api_key()

    # モデル（必要なら環境変数で差し替え）
    chat_model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    embedding_model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    tavily_key = os.environ.get("TAVILY_API_KEY")

    backend = OpenAIBackend(api_key=api_key, chat_model=chat_model, embedding_model=embedding_model)

    bot = RagWebChatBot(
        backend=backend,
        data_dir="./data/text",
        persona="default",
        web_mode="auto",
        tavily_api_key=tavily_key,
    )

    print("RAG + Web検索チャットボット（/help でコマンド、空行で終了）")

    while True:
        user = input("あなた: ")
        if user.strip() == "":
            break

        out = bot.reply(user)
        if out == "__EXIT__":
            break
        print(f"bot: {out}\n")

    print("\n---ご利用ありがとうございました！---")


if __name__ == "__main__":
    main()
