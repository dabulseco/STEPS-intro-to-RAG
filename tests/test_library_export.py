"""Tests for the Library app's full-conversation HTML export."""

import datetime
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library_ui.chat_tab import conversation_export
from rag_core.conversations import Exchange
from rag_core.library import ChatbotSettings
from utils.html_export import build_conversation_html


def exchange(pos, model):
    return Exchange(message_id=pos, position=pos, question=f"A very long first question {pos}?",
                    answer=f"Answer {pos}.", chat_model=model, created_at="2026-09-18T14:05:00")


NOW = datetime.datetime(2026, 9, 18, 14, 5, 33)


class TestConversationExport:
    def test_title_and_filename_use_library_name_and_timestamp(self):
        html_doc, filename = conversation_export(
            "Prison education status", [exchange(0, "gemma4:e4b")], ChatbotSettings(), now=NOW)
        assert filename == "Prison-education-status_2026-09-18_1405.html"
        assert "<title>Prison education status — 2026-09-18 14:05</title>" in html_doc
        assert "A very long first question" not in html_doc.split("<body")[0]

    def test_model_line_directly_below_title(self):
        html_doc, _ = conversation_export(
            "BIOL 201", [exchange(0, "gemma4:e4b")], ChatbotSettings(), now=NOW)
        after_title = html_doc.split("</h1>", 1)[1].lstrip()
        assert after_title.startswith('<p class="mb-1"><strong>Model:</strong> gemma4:e4b</p>')

    def test_several_models_listed_in_order_once(self):
        html_doc, _ = conversation_export(
            "BIOL 201", [exchange(0, "a:1"), exchange(1, "b:2"), exchange(2, "a:1")],
            ChatbotSettings(), now=NOW)
        assert "<strong>Models:</strong> a:1, b:2</p>" in html_doc

    def test_unsafe_characters_removed_from_filename(self):
        _, filename = conversation_export("Lab/Notes: v2!", [exchange(0, "m")],
                                          ChatbotSettings(), now=NOW)
        assert re.fullmatch(r"[A-Za-z0-9_-]+\.html", filename)


def test_main_app_export_unchanged_without_model():
    html_doc = build_conversation_html([("q", "a")], title="T")
    assert "Model:" not in html_doc
