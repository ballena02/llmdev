from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag_web_chatbot import Chunk, RagIndex, RagWebChatBot, WebResult


class FakeBackend:
    """
    chat() と embed_texts() を差し替えるだけでテスト可能にする。
    """
    def __init__(self) -> None:
        self.last_messages = None

    def chat(self, messages):
        self.last_messages = messages
        return "OK"

    def embed_texts(self, texts):
        # 超簡易：文字列に応じて2次元ベクトルを返す
        # "apple" が含まれるなら [1,0]、"banana" が含まれるなら [0,1]
        out = []
        for t in texts:
            t = t.lower()
            if "apple" in t:
                out.append([1.0, 0.0])
            elif "banana" in t:
                out.append([0.0, 1.0])
            else:
                out.append([0.0, 0.0])
        return out


def test_rag_index_search_prefers_relevant_chunk():
    # 2つのチャンク（埋め込みは固定）
    chunks = [
        Chunk(source="a.txt", text="about apple", embedding=[1.0, 0.0]),
        Chunk(source="b.txt", text="about banana", embedding=[0.0, 1.0]),
    ]
    index = RagIndex(chunks)
    backend = FakeBackend()

    hits = index.search("apple", embed_texts=backend.embed_texts, top_k=1)
    assert hits[0][0].source == "a.txt"


def test_chatbot_includes_rag_context_in_system_message():
    backend = FakeBackend()

    # 事前にindexを注入（ディレクトリ読み込み＆実Embedding不要）
    index = RagIndex(
        [
            Chunk(source="doc.txt", text="apple is a fruit", embedding=[1.0, 0.0]),
        ]
    )

    bot = RagWebChatBot(
        backend=backend,
        index=index,
        tavily_api_key=None,   # Web検索は無効
        web_mode="off",
    )

    ans = bot.reply("apple について教えて")
    assert ans == "OK"

    # backendに渡った messages を確認
    msgs = backend.last_messages
    assert msgs is not None
    # system persona + history + system context + user のどこかに [DOC1] が含まれているはず
    assert any(m["role"] == "system" and "[DOC1]" in m["content"] for m in msgs)


def test_web_search_mode_on_shows_web_section(monkeypatch):
    backend = FakeBackend()
    index = RagIndex([Chunk(source="doc.txt", text="apple", embedding=[1.0, 0.0])])

    # tavily_search をモック（外部通信しない）
    from rag_web_chatbot import tavily_search as real_tavily_search
    import rag_web_chatbot

    def fake_tavily_search(query, api_key, max_results=3):
        return [WebResult(title="Example", url="https://example.com", content="example content")]

    monkeypatch.setattr(rag_web_chatbot, "tavily_search", fake_tavily_search)

    bot = RagWebChatBot(
        backend=backend,
        index=index,
        tavily_api_key="dummy",
        web_mode="on",
    )

    bot.reply("最新ニュースを検索して")
    msgs = backend.last_messages
    assert any(m["role"] == "system" and "【Web検索】" in m["content"] for m in msgs)

    # 元に戻す（任意）
    monkeypatch.setattr(rag_web_chatbot, "tavily_search", real_tavily_search)


def test_history_trim_keeps_working():
    backend = FakeBackend()
    index = RagIndex([Chunk(source="doc.txt", text="apple", embedding=[1.0, 0.0])])

    bot = RagWebChatBot(
        backend=backend,
        index=index,
        max_turns=2,     # 2往復だけ保持
        web_mode="off",
    )

    for i in range(10):
        bot.reply(f"apple {i}")

    # user/assistant が最大 4件以内（2往復）
    assert len(bot.history) <= 4


def test_reset_command():
    backend = FakeBackend()
    index = RagIndex([Chunk(source="doc.txt", text="apple", embedding=[1.0, 0.0])])

    bot = RagWebChatBot(backend=backend, index=index)
    bot.reply("hello")
    assert len(bot.history) >= 2

    out = bot.reply("/reset")
    assert out == "会話履歴をリセットしました。"
    assert bot.history == []
