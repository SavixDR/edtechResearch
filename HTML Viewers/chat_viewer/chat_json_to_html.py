#!/usr/bin/env python3
"""Generate a readable, searchable HTML viewer from chat JSON/JSONL.

Supports:
- One conversation object with ``conversation_rows``
- Lists of conversations or interactions
- JSONL / NDJSON
- Loading another chat file directly from the generated HTML viewer
- Trailing commas
- ``messages`` arrays containing role/content objects
- Real and double-escaped newline/tab notation

Optional enhanced Markdown rendering:
    pip install markdown pygments
    # or: pip install mistune

Examples:
    python chat_json_to_html.py conversation.json
    python chat_json_to_html.py conversation.json -o viewer.html --open
    python chat_json_to_html.py conversations.jsonl --title "Research Chats"
    python chat_json_to_html.py malformed.json --slash-escapes

Run without a filename to use a Tkinter file picker.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

try:
    import markdown as markdown_lib
except ImportError:
    markdown_lib = None

try:
    import mistune
except ImportError:
    mistune = None


INTERACTION_CONTAINER_KEYS = ("conversation_rows", "interactions", "turns", "rows")
PROMPT_KEYS = ("prompt", "user", "question", "input", "user_message", "human")
RESPONSE_KEYS = ("response", "assistant", "answer", "output", "assistant_message", "bot")

# Standard JSON \n, \r and \t are always handled. Enable this only for malformed
# exports that literally use /n, /r and /t as control notation.
NORMALIZE_FORWARD_SLASH_ESCAPES = False


def remove_trailing_commas(text: str) -> str:
    """Remove trailing commas while respecting JSON strings."""
    output: list[str] = []
    in_string = False
    escaped = False
    i = 0

    while i < len(text):
        char = text[i]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            i += 1
            continue

        if char == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "]}":
                i += 1
                continue

        output.append(char)
        i += 1

    return "".join(output)


def parse_json_text(text: str) -> Any:
    """Parse standard JSON, tolerant JSON, or JSONL."""
    text = text.lstrip("\ufeff").strip()
    if not text:
        raise ValueError("The input file is empty.")

    errors: list[str] = []
    tolerant_text = remove_trailing_commas(text).rstrip()
    while tolerant_text.endswith(","):
        tolerant_text = tolerant_text[:-1].rstrip()

    for description, candidate in (
        ("standard JSON", text),
        ("JSON with trailing/terminal commas removed", tolerant_text),
    ):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{description}: {exc}")

    records: list[Any] = []
    jsonl_errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(remove_trailing_commas(stripped.rstrip(","))))
        except json.JSONDecodeError as exc:
            jsonl_errors.append(f"line {line_number}: {exc}")

    if records and not jsonl_errors:
        return records

    details = "\n".join(errors + jsonl_errors[:5])
    raise ValueError(f"Unable to parse the input as JSON or JSONL.\n{details}")


def read_json_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    return parse_json_text(text)


def first_value(mapping: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def looks_like_interaction(item: Any) -> bool:
    return isinstance(item, dict) and bool(
        set(item).intersection(PROMPT_KEYS) or set(item).intersection(RESPONSE_KEYS)
    )


def messages_to_rows(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert role/content messages into prompt-response interaction pairs."""
    rows: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).lower()
        content = message.get("content", "")
        timestamp = first_value(message, ("timestamp", "created_at", "time"), "")

        if role in {"user", "human"}:
            if pending is not None:
                rows.append(pending)
            pending = {
                "interactionCount": len(rows),
                "timestamp": timestamp,
                "prompt": content,
                "response": "",
                "_source_message_index": index,
            }
        elif role in {"assistant", "bot", "ai"}:
            if pending is None:
                pending = {
                    "interactionCount": len(rows),
                    "timestamp": timestamp,
                    "prompt": "",
                    "response": content,
                    "_source_message_index": index,
                }
            else:
                pending["response"] = content
            rows.append(pending)
            pending = None
        else:
            rows.append(
                {
                    "interactionCount": len(rows),
                    "timestamp": timestamp,
                    "prompt": f"[{role or 'message'}]",
                    "response": content,
                    "_source_message_index": index,
                }
            )

    if pending is not None:
        rows.append(pending)
    return rows


def get_rows(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    for key in INTERACTION_CONTAINER_KEYS:
        value = conversation.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]

    messages = conversation.get("messages")
    if isinstance(messages, list):
        return messages_to_rows(messages)

    if looks_like_interaction(conversation):
        return [conversation]
    return []


def normalize_conversations(data: Any, source_name: str) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("conversations", "chats", "data", "items"):
            value = data.get(key)
            if isinstance(value, list) and value:
                return normalize_conversations(value, source_name)

        rows = get_rows(data)
        if rows:
            conversation = dict(data)
            conversation.setdefault("chatTitle", source_name)
            if not any(key in conversation for key in INTERACTION_CONTAINER_KEYS):
                conversation["conversation_rows"] = rows
            return [conversation]

        raise ValueError(
            "No interactions were detected. Expected conversation_rows, interactions, "
            "turns, rows, messages, or prompt/response fields."
        )

    if isinstance(data, list):
        if not data:
            return []

        if all(isinstance(item, dict) and looks_like_interaction(item) for item in data):
            return [{
                "chatTitle": source_name,
                "num_interactions": len(data),
                "conversation_rows": data,
            }]

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            rows = get_rows(item)
            if not rows:
                continue
            conversation = dict(item)
            conversation.setdefault("chatTitle", f"{source_name} — Chat {index}")
            if not any(key in conversation for key in INTERACTION_CONTAINER_KEYS):
                conversation["conversation_rows"] = rows
            normalized.append(conversation)

        if normalized:
            return normalized

    raise ValueError("Unsupported JSON root structure.")


def normalize_text(value: Any) -> str:
    """Normalize CR/LF, double-escaped notation, and optional /n notation."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", "\t")
    )

    if NORMALIZE_FORWARD_SLASH_ESCAPES:
        text = (
            text.replace("/r/n", "\n")
            .replace("/n", "\n")
            .replace("/r", "\n")
            .replace("/t", "\t")
        )

    return text.replace("\x00", "")


def render_markdown(value: Any) -> str:
    """Safely render Markdown; fall back to pre-wrapped text."""
    text = normalize_text(value)

    if markdown_lib is not None:
        # Escape raw HTML first while leaving Markdown symbols intact.
        escaped = html.escape(text, quote=False)
        return markdown_lib.markdown(
            escaped,
            extensions=["fenced_code", "tables", "sane_lists", "nl2br"],
            output_format="html5",
        )

    if mistune is not None:
        renderer = mistune.create_markdown(escape=True, plugins=["table"])
        return renderer(text)

    escaped = html.escape(text, quote=False)
    return f'<div class="plain-text">{escaped}</div>'


def format_timestamp(value: Any) -> str:
    if value in (None, ""):
        return "No timestamp"

    try:
        if isinstance(value, (int, float)) or str(value).isdigit():
            numeric = float(value)
            if numeric > 10_000_000_000:
                numeric /= 1000.0
            dt = datetime.fromtimestamp(numeric, tz=timezone.utc)
        else:
            raw = str(value).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

        if ZoneInfo is not None:
            dt = dt.astimezone(ZoneInfo("Asia/Colombo"))
        else:
            dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return normalize_text(value)


def safe_json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2), quote=False)


def label_parts(row: dict[str, Any]) -> tuple[str, str]:
    label_obj = (
        row.get("llm_label")
        or row.get("label")
        or row.get("annotation")
        or row.get("classification")
    )

    if isinstance(label_obj, dict):
        label = first_value(label_obj, ("label", "name", "class", "category"), "")
        reason = first_value(
            label_obj,
            ("justification", "reason", "explanation", "rationale"),
            "",
        )
        return normalize_text(label), normalize_text(reason)

    return (normalize_text(label_obj), "") if label_obj not in (None, "") else ("", "")


def metadata_items(conversation: dict[str, Any], row_count: int) -> list[tuple[str, str]]:
    candidates = [
        ("Title", conversation.get("chatTitle") or conversation.get("title")),
        ("Topic", conversation.get("topic")),
        ("User ID", conversation.get("userId") or conversation.get("user_id")),
        ("Chat ID", conversation.get("chatId") or conversation.get("chat_id")),
        ("Started", conversation.get("chatStartTime") or conversation.get("created_at")),
        (
            "Declared interactions",
            conversation.get("num_interactions")
            or conversation.get("chatTotalInteractionCount"),
        ),
        ("Detected interactions", row_count),
    ]
    return [
        (label, normalize_text(value))
        for label, value in candidates
        if value not in (None, "")
    ]


def conversation_identifier(conversation: dict[str, Any], fallback: str) -> str:
    """Return the chat ID used to identify a conversation in the viewer."""
    return normalize_text(
        conversation.get("chat_id")
        or conversation.get("chatId")
        or conversation.get("id")
        or conversation.get("chatTitle")
        or conversation.get("title")
        or fallback
    )


def build_interaction_html(
    row: dict[str, Any],
    conversation_index: int,
    row_index: int,
    conversation_terms: str = "",
) -> tuple[str, str]:
    number = first_value(
        row, ("interactionCount", "interaction_count", "turn", "index"), row_index
    )
    timestamp = format_timestamp(
        first_value(row, ("timestamp", "created_at", "time", "date"), "")
    )
    prompt = first_value(row, PROMPT_KEYS, "")
    response = first_value(row, RESPONSE_KEYS, "")
    label, justification = label_parts(row)

    searchable = " ".join(
        [
            conversation_terms,
            normalize_text(prompt),
            normalize_text(response),
            label,
            justification,
            timestamp,
        ]
    ).lower()
    card_id = f"c{conversation_index}-i{row_index}"

    label_html = (
        f'<span class="label-chip">{html.escape(label)}</span>' if label else ""
    )
    reason_html = ""
    if justification:
        reason_html = f"""
        <details class="annotation-details">
          <summary>Label justification</summary>
          <div class="annotation-body">{render_markdown(justification)}</div>
        </details>
        """

    card = f"""
    <article class="interaction-card" id="{card_id}"
      data-conversation="{conversation_index}"
      data-search="{html.escape(searchable, quote=True)}">
      <div class="interaction-header">
        <div>
          <span class="interaction-number">Interaction {html.escape(str(number))}</span>
          <span class="timestamp">{html.escape(timestamp)}</span>
        </div>
        {label_html}
      </div>

      <section class="message user-message">
        <div class="message-role">
          <span class="role-icon">U</span><strong>User</strong>
          <button class="copy-button" type="button">Copy</button>
        </div>
        <div class="message-content">{render_markdown(prompt)}</div>
      </section>

      <section class="message assistant-message">
        <div class="message-role">
          <span class="role-icon">A</span><strong>Assistant</strong>
          <button class="copy-button" type="button">Copy</button>
        </div>
        <div class="message-content">{render_markdown(response)}</div>
      </section>

      {reason_html}

      <details class="raw-details">
        <summary>Raw interaction JSON</summary>
        <pre><code>{safe_json(row)}</code></pre>
      </details>
    </article>
    """

    nav = f"""
    <a class="nav-item" href="#{card_id}" data-conversation="{conversation_index}"
       data-target="{card_id}">
      <span>#{html.escape(str(number))}</span>
      <small>{html.escape(normalize_text(prompt)[:75] or '(empty prompt)')}</small>
    </a>
    """
    return card, nav


def build_html(
    conversations: list[dict[str, Any]], page_title: str, source_name: str
) -> str:
    sections: list[str] = []
    nav_sections: list[str] = []
    options: list[str] = []
    total_interactions = 0

    for conversation_index, conversation in enumerate(conversations):
        rows = get_rows(conversation)
        total_interactions += len(rows)
        identifier = conversation_identifier(
            conversation, f"Conversation {conversation_index + 1}"
        )
        original_title = normalize_text(
            conversation.get("chatTitle") or conversation.get("title") or ""
        )

        options.append(
            f'<option value="{conversation_index}">{html.escape(identifier)}</option>'
        )

        metadata_html = "".join(
            f"""
            <div class="metadata-item">
              <span>{html.escape(label)}</span>
              <strong>{html.escape(value)}</strong>
            </div>
            """
            for label, value in metadata_items(conversation, len(rows))
        )

        cards: list[str] = []
        nav_items: list[str] = []
        for row_index, row in enumerate(rows):
            card, nav = build_interaction_html(
                row,
                conversation_index,
                row_index,
                f"{identifier} {original_title}",
            )
            cards.append(card)
            nav_items.append(nav)

        sections.append(f"""
        <section class="conversation-section" data-conversation="{conversation_index}">
          <div class="conversation-summary">
            <div>
              <p class="eyebrow">Conversation {conversation_index + 1}</p>
              <h2>{html.escape(identifier)}</h2>
            </div>
            <span class="count-badge">{len(rows)} interactions</span>
          </div>
          <div class="metadata-grid">{metadata_html}</div>
          <div class="interaction-list">{''.join(cards)}</div>
        </section>
        """)

        nav_sections.append(f"""
        <div class="nav-conversation" data-conversation="{conversation_index}">
          <p class="sidebar-label">{html.escape(identifier)}</p>
          {''.join(nav_items)}
        </div>
        """)

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>__PAGE_TITLE__</title>
<style>
:root {
  --bg:#f4f6fa; --surface:#fff; --soft:#f8fafc; --user:#eef6ff;
  --assistant:#f7f5ff; --text:#172033; --muted:#667085; --border:#dfe4ec;
  --accent:#4f46e5; --accent-soft:#ecebff; --user-accent:#2563eb;
  --assistant-accent:#7c3aed; --shadow:0 10px 30px rgba(20,28,45,.08);
  --code:#111827; --code-text:#e5e7eb;
}
html[data-theme="dark"] {
  --bg:#0d111a; --surface:#151b26; --soft:#1b2330; --user:#142338;
  --assistant:#211b35; --text:#eef2f7; --muted:#a4adbd; --border:#2d3747;
  --accent:#9b95ff; --accent-soft:#2b2850; --user-accent:#66a3ff;
  --assistant-accent:#b497ff; --shadow:0 12px 36px rgba(0,0,0,.3);
  --code:#090d14; --code-text:#edf2f7;
}
*{box-sizing:border-box} html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}
button,input,select{font:inherit}
.app-header{position:sticky;top:0;z-index:20;padding:16px 24px;border-bottom:1px solid var(--border);background:var(--surface)}
.header-row{max-width:1500px;margin:auto;display:flex;gap:18px;align-items:center;justify-content:space-between}
.brand h1{margin:0;font-size:clamp(1.15rem,2vw,1.55rem);line-height:1.2}.brand p{margin:4px 0 0;color:var(--muted);font-size:.88rem}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:flex-end}.control,.toolbar button,.upload-button{min-height:40px;border:1px solid var(--border);border-radius:10px;background:var(--surface);color:var(--text)}
.control{padding:8px 12px}.search-control{min-width:min(340px,45vw)}.toolbar button,.upload-button{padding:8px 12px;cursor:pointer}.toolbar button:hover,.upload-button:hover{border-color:var(--accent);color:var(--accent)}.upload-button{display:inline-flex;align-items:center;font-weight:700}.file-input{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}.load-status{flex-basis:100%;margin:0;color:var(--muted);font-size:.78rem;text-align:right}.load-status.error{color:#dc2626}
.layout{max-width:1500px;margin:auto;display:grid;grid-template-columns:280px minmax(0,1fr);gap:24px;padding:24px}
.sidebar{position:sticky;top:96px;align-self:start;max-height:calc(100vh - 120px);overflow:auto;padding:18px;border:1px solid var(--border);border-radius:16px;background:var(--surface);box-shadow:var(--shadow)}
.sidebar-label{margin:0 0 10px;color:var(--muted);font-size:.78rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.nav-item{display:grid;grid-template-columns:34px minmax(0,1fr);gap:8px;align-items:start;padding:9px 8px;border-radius:9px;color:var(--text);text-decoration:none}.nav-item:hover{background:var(--soft);color:var(--accent)}
.nav-item span{color:var(--accent);font-weight:800}.nav-item small{overflow:hidden;color:var(--muted);line-height:1.35;text-overflow:ellipsis}.main-content{min-width:0}
.conversation-summary{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:16px}.conversation-summary h2{margin:0;font-size:clamp(1.45rem,3vw,2.15rem);line-height:1.15}.eyebrow{margin:0 0 5px;color:var(--accent);font-size:.78rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
.count-badge,.label-chip{display:inline-flex;align-items:center;padding:6px 10px;border-radius:999px;font-size:.8rem;font-weight:750}.count-badge{background:var(--surface);border:1px solid var(--border);white-space:nowrap}.label-chip{background:var(--accent-soft);color:var(--accent);overflow-wrap:anywhere}
.metadata-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:20px}.metadata-item{padding:12px 14px;border:1px solid var(--border);border-radius:12px;background:var(--surface);min-width:0}.metadata-item span{display:block;margin-bottom:3px;color:var(--muted);font-size:.72rem;font-weight:750;letter-spacing:.06em;text-transform:uppercase}.metadata-item strong{display:block;overflow-wrap:anywhere;font-size:.9rem}
.interaction-list{display:grid;gap:20px}.interaction-card{scroll-margin-top:100px;padding:18px;border:1px solid var(--border);border-radius:18px;background:var(--surface);box-shadow:var(--shadow)}
.interaction-card.search-hidden,.conversation-section.conversation-hidden,.nav-conversation.conversation-hidden,.nav-item.search-hidden{display:none}.interaction-header{display:flex;gap:14px;align-items:center;justify-content:space-between;margin-bottom:14px}.interaction-number{margin-right:10px;font-size:.92rem;font-weight:800}.timestamp{color:var(--muted);font-size:.78rem}
.message{margin-top:12px;border:1px solid var(--border);border-radius:14px;overflow:hidden}.user-message{background:var(--user);border-left:4px solid var(--user-accent)}.assistant-message{background:var(--assistant);border-left:4px solid var(--assistant-accent)}
.message-role{display:flex;gap:8px;align-items:center;padding:11px 14px;border-bottom:1px solid var(--border)}.role-icon{display:inline-grid;width:26px;height:26px;place-items:center;border-radius:50%;background:var(--surface);font-size:.75rem;font-weight:900}.user-message .role-icon{color:var(--user-accent)}.assistant-message .role-icon{color:var(--assistant-accent)}
.copy-button{margin-left:auto;padding:4px 9px;border:1px solid var(--border);border-radius:8px;background:var(--surface);color:var(--muted);cursor:pointer;font-size:.75rem}.message-content,.annotation-body{padding:16px;overflow-wrap:anywhere}.message-content>:first-child,.annotation-body>:first-child{margin-top:0}.message-content>:last-child,.annotation-body>:last-child{margin-bottom:0}.plain-text{white-space:pre-wrap;tab-size:4}
pre{position:relative;max-width:100%;overflow:auto;padding:14px;border-radius:10px;background:var(--code);color:var(--code-text);white-space:pre;tab-size:4}code{font-family:"Cascadia Code","SFMono-Regular",Consolas,monospace}:not(pre)>code{padding:.12em .34em;border:1px solid var(--border);border-radius:5px;background:var(--surface);color:var(--accent)}
table{width:100%;border-collapse:collapse;display:block;overflow-x:auto}th,td{padding:8px 10px;border:1px solid var(--border);text-align:left}th{background:var(--soft)}blockquote{margin-left:0;padding-left:14px;border-left:4px solid var(--accent);color:var(--muted)}img{max-width:100%}
details{margin-top:12px;border:1px solid var(--border);border-radius:10px;background:var(--soft)}summary{padding:10px 12px;cursor:pointer;color:var(--muted);font-weight:700}.raw-details pre{margin:0 12px 12px}.empty-state{display:none;padding:40px 20px;text-align:center;color:var(--muted)}.empty-state.visible{display:block}.footer{max-width:1500px;margin:auto;padding:0 24px 30px;color:var(--muted);font-size:.8rem;text-align:center}
@media(max-width:900px){.header-row{align-items:stretch;flex-direction:column}.toolbar{justify-content:flex-start}.search-control{width:100%;min-width:0}.layout{grid-template-columns:1fr;padding:16px}.sidebar{display:none}.app-header{position:static;padding:14px 16px}.interaction-card{scroll-margin-top:20px}}
@media print{.app-header,.sidebar,.copy-button,.footer{display:none!important}.layout{display:block;max-width:none;padding:0}.interaction-card{break-inside:avoid;box-shadow:none}body{background:white}}
</style>
</head>
<body>
<header class="app-header"><div class="header-row">
  <div class="brand"><h1 id="pageTitle">__PAGE_TITLE__</h1><p id="sourceSummary">__TOTAL_INTERACTIONS__ interactions from __SOURCE_NAME__</p></div>
  <div class="toolbar">
    <label class="upload-button" for="fileInput">Load chat file</label>
    <input id="fileInput" class="file-input" type="file" accept=".json,.jsonl,.ndjson,.txt,application/json,text/plain">
    <select id="conversationSelect" class="control" aria-label="Choose conversation">__OPTIONS__</select>
    <input id="searchInput" class="control search-control" type="search" placeholder="Search chat ID, prompts, responses, labels…" aria-label="Search conversations">
    <button id="clearSearch" type="button">Clear</button>
    <button id="expandAll" type="button">Expand details</button>
    <button id="collapseAll" type="button">Collapse details</button>
    <button id="themeToggle" type="button">Dark theme</button>
    <p id="loadStatus" class="load-status" role="status" aria-live="polite">Choose a file to replace the currently displayed conversations.</p>
  </div>
</div></header>
<div class="layout">
  <aside id="sidebar" class="sidebar">__NAV_SECTIONS__</aside>
  <main id="mainContent" class="main-content">__CONVERSATION_SECTIONS__<div id="emptyState" class="empty-state">No interactions match your search.</div></main>
</div>
<footer class="footer">Generated __GENERATED_AT__. This viewer is local; the chat data is not uploaded anywhere.</footer>
<script>
const select=document.getElementById("conversationSelect"),search=document.getElementById("searchInput"),empty=document.getElementById("emptyState"),themeButton=document.getElementById("themeToggle"),fileInput=document.getElementById("fileInput"),loadStatus=document.getElementById("loadStatus"),sidebar=document.getElementById("sidebar"),mainContent=document.getElementById("mainContent");
const active=()=>select.value;
function applySearch(){const q=search.value.trim().toLowerCase(),a=active();let visible=0;document.querySelectorAll(".interaction-card").forEach(card=>{const show=card.dataset.conversation===a&&(!q||card.dataset.search.includes(q));card.classList.toggle("search-hidden",!show);if(show)visible++;const nav=document.querySelector(`.nav-item[data-target="${card.id}"]`);if(nav)nav.classList.toggle("search-hidden",!show)});empty.classList.toggle("visible",visible===0)}
function updateConversation(){const a=active();document.querySelectorAll(".conversation-section,.nav-conversation").forEach(el=>el.classList.toggle("conversation-hidden",el.dataset.conversation!==a));applySearch();window.scrollTo({top:0,behavior:"smooth"})}
select.addEventListener("change",updateConversation);search.addEventListener("input",applySearch);document.getElementById("clearSearch").addEventListener("click",()=>{search.value="";applySearch();search.focus()});
document.getElementById("expandAll").addEventListener("click",()=>document.querySelectorAll(`.interaction-card[data-conversation="${active()}"] details`).forEach(x=>x.open=true));document.getElementById("collapseAll").addEventListener("click",()=>document.querySelectorAll(`.interaction-card[data-conversation="${active()}"] details`).forEach(x=>x.open=false));
document.addEventListener("click",async event=>{const button=event.target.closest(".copy-button");if(!button)return;const text=button.closest(".message").querySelector(".message-content").innerText;try{await navigator.clipboard.writeText(text);const old=button.textContent;button.textContent="Copied";setTimeout(()=>button.textContent=old,1000)}catch{button.textContent="Copy failed"}});
const interactionKeys=["conversation_rows","interactions","turns","rows"],promptKeys=["prompt","user","question","input","user_message","human"],responseKeys=["response","assistant","answer","output","assistant_message","bot"];
const esc=value=>String(value??"").replace(/[&<>\"]/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[char]));
const valueFor=(object,keys,fallback="")=>{for(const key of keys)if(object?.[key]!==undefined&&object[key]!==null)return object[key];return fallback};
const textFor=value=>{if(value===null||value===undefined)return "";const text=typeof value==="object"?JSON.stringify(value,null,2):String(value);return text.replace(/\\r\\n/g,"\n").replace(/\\r/g,"\n").replace(/\\n/g,"\n").replace(/\\t/g,"\t").replace(/\0/g,"")};
const textHtml=value=>`<div class="plain-text">${esc(textFor(value))}</div>`;
function removeTrailingCommas(text){let result="",quoted=false,escaped=false;for(let i=0;i<text.length;i++){const char=text[i];if(quoted){result+=char;if(escaped)escaped=false;else if(char==="\\")escaped=true;else if(char==='"')quoted=false;continue}if(char==='"'){quoted=true;result+=char;continue}if(char===","){let j=i+1;while(j<text.length&&/\s/.test(text[j]))j++;if("]}".includes(text[j]))continue}result+=char}return result}
function parseChatText(text){text=text.replace(/^\uFEFF/,"").trim();if(!text)throw new Error("The selected file is empty.");const tolerant=removeTrailingCommas(text).replace(/,+\s*$/,"");for(const candidate of [text,tolerant]){try{return JSON.parse(candidate)}catch{}}const records=[],errors=[];text.split(/\r?\n/).forEach((line,index)=>{if(!line.trim())return;try{records.push(JSON.parse(removeTrailingCommas(line.trim().replace(/,$/,""))))}catch{errors.push(index+1)}});if(records.length&&!errors.length)return records;throw new Error(errors.length?`Invalid JSONL on line ${errors[0]}.`:"Unable to parse this file as JSON or JSONL.")}
function looksLikeInteraction(item){return item&&typeof item==="object"&&[...promptKeys,...responseKeys].some(key=>key in item)}
function messagesToRows(messages){const rows=[];let pending=null;messages.forEach((message,index)=>{if(!message||typeof message!=="object")return;const role=String(message.role||"").toLowerCase(),content=message.content??"",timestamp=valueFor(message,["timestamp","created_at","time"]);if(["user","human"].includes(role)){if(pending)rows.push(pending);pending={interactionCount:rows.length,timestamp,prompt:content,response:"",_source_message_index:index}}else if(["assistant","bot","ai"].includes(role)){if(!pending)pending={interactionCount:rows.length,timestamp,prompt:"",response:content,_source_message_index:index};else pending.response=content;rows.push(pending);pending=null}else rows.push({interactionCount:rows.length,timestamp,prompt:`[${role||"message"}]`,response:content,_source_message_index:index})});if(pending)rows.push(pending);return rows}
function rowsFor(conversation){for(const key of interactionKeys)if(Array.isArray(conversation?.[key]))return conversation[key].filter(row=>row&&typeof row==="object");if(Array.isArray(conversation?.messages))return messagesToRows(conversation.messages);return looksLikeInteraction(conversation)?[conversation]:[]}
function normalizeLoaded(data,source){if(data&&typeof data==="object"&&!Array.isArray(data)){for(const key of ["conversations","chats","data","items"])if(Array.isArray(data[key])&&data[key].length)return normalizeLoaded(data[key],source);const rows=rowsFor(data);if(rows.length)return [{...data,chatTitle:data.chatTitle||source,conversation_rows:rows}];throw new Error("No interactions were detected in this file.")}if(Array.isArray(data)){if(!data.length)return [];if(data.every(looksLikeInteraction))return [{chatTitle:source,num_interactions:data.length,conversation_rows:data}];const conversations=data.filter(item=>item&&typeof item==="object"&&rowsFor(item).length).map((item,index)=>({...item,chatTitle:item.chatTitle||item.title||`${source} — Chat ${index+1}`,conversation_rows:rowsFor(item)}));if(conversations.length)return conversations}throw new Error("Unsupported JSON root structure.")}
function timestampFor(value){if(value===null||value===undefined||value==="")return "No timestamp";const numeric=typeof value==="number"||/^\d+$/.test(String(value))?Number(value):null,date=numeric!==null?new Date(numeric>1e10?numeric:numeric*1000):new Date(value);return Number.isNaN(date.valueOf())?textFor(value):date.toLocaleString()}
function labelFor(row){const label=row.llm_label??row.label??row.annotation??row.classification;if(label&&typeof label==="object")return [valueFor(label,["label","name","class","category"]),valueFor(label,["justification","reason","explanation","rationale"])];return [label??"",""]}
function renderLoaded(conversations,fileName){
  let total=0;
  select.innerHTML="";
  sidebar.innerHTML="";
  mainContent.querySelectorAll(".conversation-section").forEach(node=>node.remove());
  conversations.forEach((conversation,ci)=>{
    const rows=rowsFor(conversation);
    const title=textFor(conversation.chatTitle||conversation.title||"");
    const identifier=textFor(conversation.chat_id||conversation.chatId||conversation.id||title||`Conversation ${ci+1}`);
    total+=rows.length;
    select.insertAdjacentHTML("beforeend",`<option value="${ci}">${esc(identifier)}</option>`);
    const nav=document.createElement("div");
    nav.className="nav-conversation";
    nav.dataset.conversation=ci;
    nav.innerHTML=`<p class="sidebar-label">${esc(identifier)}</p>`;
    const section=document.createElement("section");
    section.className="conversation-section";
    section.dataset.conversation=ci;
    const metadata=[["Title",title],["Topic",conversation.topic],["User ID",conversation.userId??conversation.user_id],["Chat ID",conversation.chatId??conversation.chat_id],["Started",conversation.chatStartTime??conversation.created_at],["Declared interactions",conversation.num_interactions??conversation.chatTotalInteractionCount],["Detected interactions",rows.length]].filter(([,value])=>value!==null&&value!==undefined&&value!=="").map(([label,value])=>`<div class="metadata-item"><span>${esc(label)}</span><strong>${esc(textFor(value))}</strong></div>`).join("");
    section.innerHTML=`<div class="conversation-summary"><div><p class="eyebrow">Conversation ${ci+1}</p><h2>${esc(identifier)}</h2></div><span class="count-badge">${rows.length} interactions</span></div><div class="metadata-grid">${metadata}</div><div class="interaction-list"></div>`;
    const list=section.querySelector(".interaction-list");
    rows.forEach((row,ri)=>{
      const number=valueFor(row,["interactionCount","interaction_count","turn","index"],ri),prompt=valueFor(row,promptKeys),response=valueFor(row,responseKeys),[label,reason]=labelFor(row),timestamp=timestampFor(valueFor(row,["timestamp","created_at","time","date"])),id=`c${ci}-i${ri}`,searchable=[identifier,title,textFor(prompt),textFor(response),textFor(label),textFor(reason),timestamp].join(" ").toLowerCase();
      list.insertAdjacentHTML("beforeend",`<article class="interaction-card" id="${id}" data-conversation="${ci}" data-search="${esc(searchable)}"><div class="interaction-header"><div><span class="interaction-number">Interaction ${esc(number)}</span><span class="timestamp">${esc(timestamp)}</span></div>${label?`<span class="label-chip">${esc(label)}</span>`:""}</div><section class="message user-message"><div class="message-role"><span class="role-icon">U</span><strong>User</strong><button class="copy-button" type="button">Copy</button></div><div class="message-content">${textHtml(prompt)}</div></section><section class="message assistant-message"><div class="message-role"><span class="role-icon">A</span><strong>Assistant</strong><button class="copy-button" type="button">Copy</button></div><div class="message-content">${textHtml(response)}</div></section>${reason?`<details class="annotation-details"><summary>Label justification</summary><div class="annotation-body">${textHtml(reason)}</div></details>`:""}<details class="raw-details"><summary>Raw interaction JSON</summary><pre><code>${esc(JSON.stringify(row,null,2))}</code></pre></details></article>`);
      nav.insertAdjacentHTML("beforeend",`<a class="nav-item" href="#${id}" data-conversation="${ci}" data-target="${id}"><span>#${esc(number)}</span><small>${esc(textFor(prompt).slice(0,75)||"(empty prompt)")}</small></a>`)
    });
    sidebar.append(nav);
    mainContent.insertBefore(section,empty)
  });
  document.getElementById("pageTitle").textContent=conversations.length===1?textFor(conversations[0].chat_id||conversations[0].chatId||conversations[0].id||conversations[0].chatTitle||conversations[0].title||fileName):`${fileName.replace(/\.[^.]+$/,"")} — Chat Viewer`;
  document.getElementById("sourceSummary").textContent=`${total} interactions from ${fileName}`;
  search.value="";
  select.value="0";
  updateConversation()
}
fileInput.addEventListener("change",async()=>{const file=fileInput.files?.[0];if(!file)return;loadStatus.classList.remove("error");loadStatus.textContent=`Loading ${file.name}…`;try{const data=parseChatText(await file.text()),source=file.name.replace(/\.[^.]+$/,"")||file.name,conversations=normalizeLoaded(data,source);if(!conversations.length)throw new Error("The selected file contains no conversations.");renderLoaded(conversations,file.name);loadStatus.textContent=`Loaded ${conversations.length} conversation${conversations.length===1?"":"s"} from ${file.name}. Data stayed in this browser.`}catch(error){loadStatus.classList.add("error");loadStatus.textContent=`Could not load ${file.name}: ${error.message}`}finally{fileInput.value=""}});
function setTheme(theme){document.documentElement.dataset.theme=theme;localStorage.setItem("chatViewerTheme",theme);themeButton.textContent=theme==="dark"?"Light theme":"Dark theme"}const stored=localStorage.getItem("chatViewerTheme"),preferred=window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";setTheme(stored||preferred);themeButton.addEventListener("click",()=>setTheme(document.documentElement.dataset.theme==="dark"?"light":"dark"));updateConversation();
</script>
</body></html>'''

    replacements = {
        "__PAGE_TITLE__": html.escape(page_title),
        "__TOTAL_INTERACTIONS__": str(total_interactions),
        "__SOURCE_NAME__": html.escape(source_name),
        "__OPTIONS__": "\n".join(options),
        "__NAV_SECTIONS__": "\n".join(nav_sections),
        "__CONVERSATION_SECTIONS__": "\n".join(sections),
        "__GENERATED_AT__": html.escape(generated_at),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def choose_input_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select chat JSON or JSONL file",
        filetypes=[
            ("JSON files", "*.json *.jsonl *.ndjson"),
            ("Text files", "*.txt"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def choose_output_file(default_path: Path) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return default_path

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.asksaveasfilename(
        title="Save chat viewer HTML",
        initialdir=str(default_path.parent),
        initialfile=default_path.name,
        defaultextension=".html",
        filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a searchable HTML viewer from chat JSON/JSONL."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input JSON/JSONL/TXT")
    parser.add_argument("-o", "--output", type=Path, help="Output HTML path")
    parser.add_argument("--title", help="Viewer page title")
    parser.add_argument("--open", action="store_true", help="Open HTML in browser")
    parser.add_argument(
        "--gui-save", action="store_true", help="Use a Tkinter Save As dialog"
    )
    parser.add_argument(
        "--slash-escapes",
        action="store_true",
        help="Also convert literal /n, /r and /t notation (disabled by default to protect URLs and paths)",
    )
    return parser.parse_args()


def main() -> int:
    global NORMALIZE_FORWARD_SLASH_ESCAPES
    args = parse_args()
    NORMALIZE_FORWARD_SLASH_ESCAPES = args.slash_escapes
    input_path = args.input
    used_gui = False

    if input_path is None:
        input_path = choose_input_file()
        used_gui = True
        if input_path is None:
            print("No input file selected.", file=sys.stderr)
            return 1

    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    try:
        data = read_json_file(input_path)
        conversations = normalize_conversations(data, input_path.stem)
    except Exception as exc:
        print(f"Could not process the input file:\n{exc}", file=sys.stderr)
        return 1

    output_path = (
        args.output.expanduser()
        if args.output
        else input_path.with_name(f"{input_path.stem}_viewer.html")
    )

    if args.gui_save or used_gui:
        selected = choose_output_file(output_path)
        if selected is None:
            print("No output file selected.", file=sys.stderr)
            return 1
        output_path = selected

    output_path = output_path.resolve()
    page_title = args.title or (
        conversation_identifier(conversations[0], input_path.stem)
        if len(conversations) == 1
        else f"{input_path.stem} — Chat Viewer"
    )

    output_path.write_text(
        build_html(conversations, page_title, input_path.name), encoding="utf-8"
    )

    if markdown_lib is not None:
        markdown_status = "enhanced with Python-Markdown"
    elif mistune is not None:
        markdown_status = "enhanced with Mistune"
    else:
        markdown_status = "plain-text fallback"
    print(f"Created: {output_path}")
    print(f"Conversations: {len(conversations)}")
    print(f"Interactions: {sum(len(get_rows(c)) for c in conversations)}")
    print(f"Markdown rendering: {markdown_status}")

    if args.open or used_gui:
        webbrowser.open(output_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
