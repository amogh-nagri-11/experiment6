"""
intent.py - LLM intent labelling for UI graphs (Figma side now, DOM side later - same dict shape).

Each node gets:
    role        one of ROLES (fixed vocabulary, shared by both graphs so they're comparable)
    intent      short free-text purpose, e.g. "primary action: submit login form"
    confidence  0..1
    role_source "llm" | "rule"   (rule = the heuristic role kept as fallback)

Backends (pluggable):
    copilot   GitHub Copilot SDK (pip install microsoft-copilot-sdk), uses your Copilot login
    deepseek  DeepSeek OpenAI-compatible API (env DEEPSEEK_API_KEY)
    rules     no LLM - hardcoded RULE_INTENTS templates (+ optional intent map file)
    pkg.module:ClassName   any class with async start()/complete(system, user)/close()

Labels are cached in .intent_cache.json keyed by (backend, model, node payload), so re-runs on an
unchanged design cost nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Protocol

ROLES = [
    "screen", "navigation", "header", "sidebar", "footer", "main", "modal", "toolbar",
    "tabs", "breadcrumb", "menu", "form", "input", "button", "link", "table", "list",
    "list_item", "card", "chart", "image", "icon", "badge",
    "heading", "body_text", "label", "container", "other",
]

SYSTEM_PROMPT = f"""You label elements of a UI design tree by their user-facing purpose.
For every node you receive, return:
- role: exactly one of {ROLES}
- intent: <= 12 words describing what the element is FOR from the user's point of view
  (e.g. "primary action: save report", "filter results by date", "shows unread count").
  Be specific to this screen; never restate the layer name.
- confidence: 0..1

Use all evidence: layer name, component name and variant, text content, layout, the position box
`rel` = [x, y, w, h] as fractions of the screen, parent, and children. Layer names may be
meaningless ("Frame 12") - then rely on structure, text and position. `hint` is a keyword
heuristic and may be wrong.

Reply with JSON only, no prose:
{{"labels": [{{"id": "...", "role": "...", "intent": "...", "confidence": 0.0}}]}}
Include every id you were given, exactly once."""


# --------------------------------------------------------------------------- backends
class Backend(Protocol):
    name: str
    model: str
    async def start(self) -> None: ...
    async def complete(self, system: str, user: str) -> str: ...
    async def close(self) -> None: ...


class CopilotBackend:
    """GitHub Copilot SDK. One fresh, tool-less session per request so context never grows."""
    name = "copilot"

    def __init__(self, model: str | None = None, timeout: float = 180):
        self.model = model or os.environ.get("COPILOT_MODEL", "gpt-4.1")
        self.timeout = timeout
        self._client = None

    async def start(self) -> None:
        from copilot import CopilotClient          # pip install microsoft-copilot-sdk
        self._client = CopilotClient(mode="empty")  # no coding-agent defaults / env context
        await self._client.start()

    async def complete(self, system: str, user: str) -> str:
        session = await self._client.create_session(
            model=self.model,
            system_message={"mode": "replace", "content": system},
            tools={"available_tools": []},          # pure LLM call: no file/shell tools
        )
        try:
            event = await session.send_and_wait(user, timeout=self.timeout)
            return getattr(event.data, "content", "") if event else ""
        finally:
            await session.disconnect()

    async def close(self) -> None:
        if self._client:
            await self._client.stop()


class OpenAICompatBackend:
    """Any OpenAI-compatible /chat/completions endpoint (DeepSeek, Ollama, vLLM, ...)."""
    name = "openai-compat"

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: float = 180,
                 json_mode: bool = True):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.timeout, self.json_mode = timeout, json_mode

    async def start(self) -> None:
        pass

    async def complete(self, system: str, user: str) -> str:
        import requests

        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        def call():
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return await asyncio.to_thread(call)

    async def close(self) -> None:
        pass


def _deepseek(model: str | None = None) -> Backend:
    key = os.environ.get("DEEPSEEK_API_KEY") or exit("Set DEEPSEEK_API_KEY")
    b = OpenAICompatBackend("https://api.deepseek.com", key, model or "deepseek-chat")
    b.name = "deepseek"
    return b


BACKENDS: dict[str, Callable[..., Backend]] = {
    "copilot": lambda model=None: CopilotBackend(model),
    "deepseek": _deepseek,
}


def register_backend(name: str, factory: Callable[..., Backend]) -> None:
    BACKENDS[name] = factory


def get_backend(spec: str, model: str | None = None) -> Backend:
    """'copilot' | 'deepseek' | 'my_pkg.my_module:MyBackend'"""
    if spec in BACKENDS:
        return BACKENDS[spec](model)
    if ":" in spec:
        mod, cls = spec.split(":", 1)
        return getattr(importlib.import_module(mod), cls)(model)
    raise ValueError(f"Unknown backend {spec!r}. Known: {list(BACKENDS)} or module:Class")


# --------------------------------------------------------------------------- labelling
def _records(screen: dict) -> list[tuple[dict, dict]]:
    """(graph node, compact LLM payload) for every node in a screen, depth-first."""
    out: list[tuple[dict, dict]] = []

    def walk(n: dict, parent: dict | None, depth: int):
        kids = n.get("children", [])
        rec = {
            "id": n["id"],
            "parent": parent["id"] if parent else None,
            "parent_name": parent["name"] if parent else None,
            "depth": depth,
            "type": n.get("figma_type"),
            "name": n.get("name"),
            "component": n.get("component"),
            "variant": n.get("variant"),
            "text": n.get("text"),
            "layout": n.get("layout"),
            "rel": n.get("rel"),
            "children": [c.get("component") or c.get("name") for c in kids[:8]] or None,
            "n_children": len(kids) or None,
            "hint": n.get("role") if n.get("role") not in ("container", "component") else None,
        }
        out.append((n, {k: v for k, v in rec.items() if v is not None}))
        for c in kids:
            walk(c, n, depth + 1)
    walk(screen, None, 0)
    return out


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1]) if start != -1 else {}


class IntentCache:
    def __init__(self, path: Path = Path(".intent_cache.json")):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def key(backend: Backend, rec: dict) -> str:
        blob = json.dumps([backend.name, backend.model, rec], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data))


async def label_graph(graph: dict, backend: Backend, chunk_size: int = 40,
                      concurrency: int = 3, cache: IntentCache | None = None,
                      intent_map: "IntentMap | None" = None,
                      log: Callable[[str], None] = print) -> dict:
    """LLM labelling. Nodes matched by `intent_map` are fixed up-front and never sent."""
    cache = cache or IntentCache()
    sem = asyncio.Semaphore(concurrency)
    await backend.start()
    try:
        for screen in graph["screens"]:
            recs = _records(screen)
            todo = []
            for node, rec in recs:
                fixed = intent_map.match(node) if intent_map else None
                if fixed:
                    _apply(node, fixed, "map")
                    continue
                hit = cache.data.get(IntentCache.key(backend, rec))
                if hit:
                    _apply(node, hit)
                else:
                    node.setdefault("rule_role", node.get("role"))
                    todo.append((node, rec))
            log(f"[intent] {screen['name']}: {len(recs)} nodes, {len(todo)} to label "
                f"({(len(todo) + chunk_size - 1) // chunk_size} request(s))")

            outline = [c.get("name") for c in screen.get("children", [])]
            chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]

            async def run(chunk):
                prompt = json.dumps({"screen": screen["name"], "screen_outline": outline,
                                     "nodes": [r for _, r in chunk]}, ensure_ascii=False)
                async with sem:
                    try:
                        raw = await backend.complete(SYSTEM_PROMPT, prompt)
                        labels = {l.get("id"): l for l in _parse_json(raw).get("labels", [])}
                    except Exception as e:                      # keep rule roles on failure
                        log(f"[intent] chunk failed ({type(e).__name__}: {e}); keeping rule roles")
                        labels = {}
                for node, rec in chunk:
                    lab = labels.get(node["id"])
                    if not lab:
                        _apply(node, rule_label(node), "rule")
                        continue
                    lab = {"role": lab.get("role") if lab.get("role") in ROLES else "other",
                           "intent": str(lab.get("intent", ""))[:120],
                           "confidence": float(lab.get("confidence", 0) or 0)}
                    _apply(node, lab)
                    cache.data[IntentCache.key(backend, rec)] = lab

            await asyncio.gather(*(run(c) for c in chunks))
            screen["role"] = "screen"                            # root stays a screen
            cache.save()
    finally:
        await backend.close()
    return graph


def _apply(node: dict, lab: dict, source: str = "llm") -> None:
    node.setdefault("rule_role", node.get("role"))
    node["role"] = lab["role"]
    node["intent"] = lab["intent"]
    node["confidence"] = lab["confidence"]
    node["role_source"] = source


# --------------------------------------------------------------------------- hardcoded intents
# Template per role; {text} {component} {name} {parent} are filled from the node.
RULE_INTENTS = {
    "screen": "screen: {name}",
    "navigation": "navigate between app areas",
    "header": "page header",
    "sidebar": "switch between sections",
    "footer": "page footer / secondary links",
    "main": "primary content of {parent}",
    "modal": "focused task in an overlay: {name}",
    "toolbar": "actions and filters for the current view",
    "tabs": "switch between views",
    "breadcrumb": "show location and go up a level",
    "menu": "choose from a list of actions",
    "form": "collect user input: {name}",
    "input": "enter or choose a value: {label}",
    "button": "action: {label}",
    "link": "go to: {label}",
    "table": "show records in rows and columns",
    "list": "show a collection of items",
    "list_item": "item: {label}",
    "card": "summary of one item: {label}",
    "chart": "visualise data",
    "image": "visual: {name}",
    "icon": "icon: {name}",
    "badge": "status or count indicator",
    "heading": "title: {text}",
    "body_text": "content text",
    "label": "label: {text}",
    "container": "group of {n} elements",
    "other": "{label}",
}


def _rule_role(node: dict) -> str:
    """Map figma_graph's heuristic role onto the shared ROLES vocabulary."""
    r = node.get("role")
    if r in ROLES:
        return r
    if r == "text":
        t, rel = node.get("text") or "", node.get("rel") or [0, 0, 0, 0]
        if len(t.split()) <= 4 and rel[3] > 0.03:
            return "heading"
        return "label" if len(t.split()) <= 4 else "body_text"
    if r == "component":
        return "container" if node.get("children") else "other"
    return "other"


class _SafeDict(dict):
    def __missing__(self, k):
        return ""


def _fill(template: str, node: dict, parent: dict | None = None) -> str:
    label = node.get("text") or node.get("component") or node.get("name") or ""
    out = template.format_map(_SafeDict(
        text=node.get("text") or "", component=node.get("component") or "",
        name=node.get("name") or "", label=label, n=len(node.get("children", [])),
        parent=(parent or {}).get("name", "screen")))
    return re.sub(r"[:\s]+$", "", out).strip()[:120]


def rule_label(node: dict, parent: dict | None = None) -> dict:
    role = _rule_role(node)
    return {"role": role, "intent": _fill(RULE_INTENTS.get(role, "{label}"), node, parent),
            "confidence": 0.5}


class IntentMap:
    """User-supplied hardcoded intents (JSON, or YAML if PyYAML is installed). First match wins.

    [
      {"match": {"component": "^Button$", "variant.Variant": "Primary"},
       "role": "button", "intent": "primary action: {text}"},
      {"match": {"name": "(?i)side ?nav"}, "role": "sidebar", "intent": "switch app sections"},
      {"match": {"id": "12:345"}, "role": "chart", "intent": "revenue trend for last 30 days"}
    ]
    match keys: id, name, component, text, figma_type, role (heuristic), variant.<prop>
    values are regexes (re.search); all keys must match. role/intent templates may use
    {text} {component} {name} {label} {parent} {n}.
    """

    def __init__(self, rules: list[dict]):
        self.rules = []
        for r in rules:
            if r.get("role") and r["role"] not in ROLES:
                raise ValueError(f"intent map: role {r['role']!r} not in ROLES")
            self.rules.append((r, {k: re.compile(str(v)) for k, v in r.get("match", {}).items()}))

    @classmethod
    def load(cls, path: str | Path) -> "IntentMap":
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            import yaml
            return cls(yaml.safe_load(text) or [])
        return cls(json.loads(text))

    @staticmethod
    def _field(node: dict, key: str):
        if key.startswith("variant."):
            return (node.get("variant") or {}).get(key[8:])
        return node.get(key)

    def match(self, node: dict, parent: dict | None = None) -> dict | None:
        for rule, pats in self.rules:
            if all(self._field(node, k) is not None and p.search(str(self._field(node, k)))
                   for k, p in pats.items()):
                role = rule.get("role") or _rule_role(node)
                tmpl = rule.get("intent") or RULE_INTENTS.get(role, "{label}")
                return {"role": role, "intent": _fill(tmpl, node, parent),
                        "confidence": float(rule.get("confidence", 1.0))}
        return None


def label_rules(graph: dict, intent_map: IntentMap | None = None) -> dict:
    """No LLM: intent map first, then built-in RULE_INTENTS templates."""
    def walk(n: dict, parent: dict | None):
        kids = n.get("children", [])
        fixed = intent_map.match(n, parent) if intent_map else None
        if fixed:
            _apply(n, fixed, "map")
        else:
            _apply(n, rule_label(n, parent), "rule")
        for c in kids:
            walk(c, n)
    for screen in graph["screens"]:
        walk(screen, None)
        screen["role"] = "screen"
    return graph