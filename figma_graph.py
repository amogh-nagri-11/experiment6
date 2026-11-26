#!/usr/bin/env python3
"""
figma_graph.py - Build a high-level, intent-oriented UI graph from a Figma design URL.

    screen -> navigation / sidebar / main / footer -> components -> nested components

Usage:
    cp .env.example .env      # fill FIGMA_TOKEN + FIGMA_URL (everything is configured there)
    python figma_graph.py     # or: python figma_graph.py --env other.env

Output: a flat semantic index (see semantics.py) for element-to-element matching with the DOM.

Notes:
  * One Tier-1 API call per (file, node) - results are cached in .figma_cache/ because
    Starter-plan limits are tiny. Pass a node-id in the URL to fetch just one screen.
  * Leaf = a component INSTANCE (unless it contains nested instances), a TEXT, or an image.
    Decorative vectors/shapes are dropped; auto-named wrappers ("Frame 12") with one child
    are collapsed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

API = "https://api.figma.com/v1"
CACHE_DIR = Path(".figma_cache")

DECORATIVE = {"VECTOR", "LINE", "ELLIPSE", "STAR", "REGULAR_POLYGON",
              "BOOLEAN_OPERATION", "RECTANGLE", "SLICE"}
SCREEN_TYPES = {"FRAME", "COMPONENT", "COMPONENT_SET", "INSTANCE"}

# Figma's auto-generated layer names carry no intent.
GENERIC_NAME = re.compile(
    r"^(frame|group|rectangle|vector|ellipse|auto ?layout|container|wrapper|section|div|layer)\s*\d*$",
    re.I)

# Name-based intent. First match wins.
ROLE_PATTERNS = [
    ("list_item",  r"\b(nav ?item|menu ?item|list ?item|option)\b"),
    ("sidebar",    r"\b(side ?bar|side ?nav|side ?panel|drawer|rail)\b"),
    ("navigation", r"\b(nav|navbar|navigation|top ?bar|app ?bar|menu ?bar|breadcrumbs?)\b"),
    ("header",     r"\b(header|masthead)\b"),
    ("footer",     r"\bfooter\b"),
    ("modal",      r"\b(modal|dialog|popup|pop-up|overlay)\b"),
    ("toolbar",    r"\b(toolbar|action ?bar|filters?)\b"),
    ("main",       r"\b(main|content|body)\b"),
    ("tabs",       r"\btabs?\b"),
    ("table",      r"\b(table|data ?grid|grid)\b"),
    ("list",       r"\b(list|feed)\b"),
    ("card",       r"\b(card|tile)\b"),
    ("form",       r"\bform\b"),
    ("button",     r"\b(button|btn|cta)\b"),
    ("input",      r"\b(input|text ?field|search|select|dropdown|checkbox|radio|toggle|switch)\b"),
    ("icon",       r"\b(icon|ico)\b"),
    ("image",      r"\b(image|img|avatar|logo|thumbnail|illustration)\b"),
]
REGION_ROLES = {"navigation", "header", "sidebar", "footer", "main", "modal", "toolbar"}


# --------------------------------------------------------------------------- model
@dataclass
class GNode:
    id: str
    name: str
    role: str                      # screen | navigation | sidebar | main | component | container | text | image ...
    figma_type: str
    bbox: dict | None = None       # absolute, from Figma
    rel: list | None = None        # [x, y, w, h] normalised to the screen (0..1) - for DOM matching
    component: str | None = None   # component-set / component name for instances
    variant: dict | None = None    # variant + component properties
    layout: str | None = None      # row | column | None
    text: str | None = None        # text content (TEXT) or label inside a leaf instance
    children: list["GNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, [], {}) and k != "children"}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class Ctx:
    components: dict
    component_sets: dict
    max_depth: int = 12


# --------------------------------------------------------------------------- fetch
def parse_figma_url(url: str) -> tuple[str, str | None]:
    """Accepts a full Figma URL or a bare file key (optionally 'KEY:12-345' / 'KEY?node-id=12-345')."""
    if not url.startswith("http"):
        parts = re.split(r"\?node-id=|:", url, maxsplit=1)
        key, node = parts[0], (parts[1] if len(parts) > 1 else None)
        return key, node.replace("-", ":") if node else None
    u = urlparse(url)
    parts = [p for p in u.path.split("/") if p]
    if len(parts) < 2 or parts[0] not in ("design", "file", "proto"):
        raise ValueError(f"Not a Figma design URL: {url}")
    key = parts[1]
    if len(parts) >= 4 and parts[2] == "branch":   # /design/:key/branch/:branchKey/:name
        key = parts[3]
    node_id = parse_qs(u.query).get("node-id", [None])[0]
    return key, node_id.replace("-", ":") if node_id else None


def fetch_payload(file_key: str, node_id: str | None, token: str, refresh: bool = False) -> dict:
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{file_key}_{(node_id or 'root').replace(':', '-')}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())

    headers = {"X-Figma-Token": token}
    if node_id:
        url, params = f"{API}/files/{file_key}/nodes", {"ids": node_id}
    else:
        url, params = f"{API}/files/{file_key}", {}

    r = requests.get(url, headers=headers, params=params, timeout=180)
    if r.status_code == 429:
        sys.exit(f"429 rate limited. Retry-After={r.headers.get('Retry-After')}s "
                 f"plan={r.headers.get('X-Figma-Plan-Tier')} "
                 f"limit-type={r.headers.get('X-Figma-Rate-Limit-Type')}")
    r.raise_for_status()
    data = r.json()

    if node_id:
        entry = (data.get("nodes") or {}).get(node_id)
        if not entry:
            sys.exit(f"Node {node_id} not found in file {file_key}")
        src = entry
    else:
        src = data
    payload = {
        "fileName": data.get("name"),
        "lastModified": data.get("lastModified"),
        "document": src["document"],
        "components": src.get("components", {}),
        "componentSets": src.get("componentSets", {}),
    }
    cache.write_text(json.dumps(payload))
    return payload


# --------------------------------------------------------------------------- helpers
def visible(n: dict) -> bool:
    return n.get("visible", True) and n.get("opacity", 1) > 0.01


def has_image_fill(n: dict) -> bool:
    return any(f.get("type") == "IMAGE" and f.get("visible", True) for f in n.get("fills", []) or [])


def name_role(name: str | None) -> str | None:
    if not name:
        return None
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)          # NavItem -> Nav Item
    s = re.sub(r"[_\-/]+", " ", s.lower())
    for role, pat in ROLE_PATTERNS:
        if re.search(pat, s):
            return role
    return None


def is_generic(name: str) -> bool:
    return bool(GENERIC_NAME.match(name.strip()))


def layout_of(n: dict) -> str | None:
    return {"HORIZONTAL": "row", "VERTICAL": "column"}.get(n.get("layoutMode"))


def reading_order(nodes: list[GNode]) -> list[GNode]:
    def key(g: GNode):
        b = g.bbox or {}
        return (round(b.get("y", 0) / 16), b.get("x", 0))
    return sorted(nodes, key=key)


def collect_text(n: dict, limit: int = 80) -> str | None:
    out: list[str] = []

    def walk(x):
        if not visible(x):
            return
        if x.get("type") == "TEXT" and (x.get("characters") or "").strip():
            out.append(x["characters"].strip())
        for c in x.get("children", []) or []:
            walk(c)
    walk(n)
    s = " ".join(out)
    return s[:limit] if s else None


def resolve_component(n: dict, ctx: Ctx) -> tuple[str, dict | None]:
    comp = ctx.components.get(n.get("componentId"), {})
    name = comp.get("name") or n.get("name")
    variant: dict = {}
    set_id = comp.get("componentSetId")
    if set_id and set_id in ctx.component_sets:
        # variant components are named "Size=Large, State=Hover"
        for part in (comp.get("name") or "").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                variant[k.strip()] = v.strip()
        name = ctx.component_sets[set_id].get("name", name)
    for k, v in (n.get("componentProperties") or {}).items():
        if v.get("type") in ("VARIANT", "BOOLEAN", "TEXT"):
            variant[k.split("#")[0]] = v.get("value")
    return name, variant or None


# --------------------------------------------------------------------------- build
def build(n: dict, ctx: Ctx, depth: int = 0) -> list[GNode]:
    """Return 0..n graph nodes for a Figma node (0 = dropped, >1 = wrapper flattened)."""
    if not visible(n) or depth > ctx.max_depth:
        return []
    t = n.get("type")
    base = dict(id=n["id"], name=n.get("name", ""), figma_type=t, bbox=n.get("absoluteBoundingBox"))

    if t == "TEXT":
        txt = (n.get("characters") or "").strip()
        return [GNode(role="text", text=txt[:80], **base)] if txt else []

    if t in DECORATIVE:
        return [GNode(role="image", **base)] if has_image_fill(n) else []

    if t == "INSTANCE":
        comp, variant = resolve_component(n, ctx)
        kids = nested_instances(n, ctx, depth + 1)
        return [GNode(role=name_role(n.get("name")) or name_role(comp) or "component",
                      component=comp, variant=variant, layout=layout_of(n),
                      text=None if kids else collect_text(n),
                      children=kids, **base)]

    # FRAME / GROUP / SECTION / COMPONENT ... -> container
    kids: list[GNode] = []
    for c in n.get("children", []) or []:
        kids.extend(build(c, ctx, depth + 1))
    kids = reading_order(kids)

    role = name_role(n.get("name"))
    if not kids:
        # empty frame with an image fill is an image; otherwise purely decorative
        return [GNode(role="image", **base)] if has_image_fill(n) else []
    if role is None and is_generic(n.get("name", "")) and len(kids) == 1:
        return kids                                   # pass-through wrapper
    return [GNode(role=role or "container", layout=layout_of(n), children=kids, **base)]


def nested_instances(n: dict, ctx: Ctx, depth: int) -> list[GNode]:
    """Inside an instance, only surface nested component instances (keeps leaves meaningful)."""
    out: list[GNode] = []
    for c in n.get("children", []) or []:
        if not visible(c):
            continue
        if c.get("type") == "INSTANCE":
            out.extend(build(c, ctx, depth))
        elif c.get("children"):
            out.extend(nested_instances(c, ctx, depth + 1))
    return reading_order(out)


def infer_regions(screen: GNode) -> None:
    """Geometry fallback for top-level regions the designer didn't name."""
    sb = screen.bbox
    if not sb or not sb.get("width") or not sb.get("height"):
        return
    W, H, sx, sy = sb["width"], sb["height"], sb["x"], sb["y"]
    unnamed = []
    for c in screen.children:
        if c.role in REGION_ROLES or not c.bbox:
            continue
        b = c.bbox
        x, y, w, h = b["x"] - sx, b["y"] - sy, b["width"], b["height"]
        if y < 0.05 * H and w > 0.8 * W and h < 0.2 * H:
            c.role = "navigation"
        elif y + h > 0.95 * H and w > 0.8 * W and h < 0.2 * H:
            c.role = "footer"
        elif x < 0.05 * W and h > 0.6 * H and w < 0.35 * W:
            c.role = "sidebar"
        elif c.role in ("container", "component"):
            unnamed.append(c)
    if not any(c.role == "main" for c in screen.children) and unnamed:
        big = max(unnamed, key=lambda g: g.bbox["width"] * g.bbox["height"])
        if big.bbox["width"] * big.bbox["height"] > 0.25 * W * H:
            big.role = "main"


def add_relative_boxes(node: GNode, sb: dict) -> None:
    if node.bbox and sb.get("width") and sb.get("height"):
        b = node.bbox
        node.rel = [round((b["x"] - sb["x"]) / sb["width"], 4), round((b["y"] - sb["y"]) / sb["height"], 4),
                    round(b["width"] / sb["width"], 4), round(b["height"] / sb["height"], 4)]
    for c in node.children:
        add_relative_boxes(c, sb)


def make_screen(frame: dict, ctx: Ctx) -> GNode:
    kids: list[GNode] = []
    for c in frame.get("children", []) or []:
        kids.extend(build(c, ctx, 1))
    screen = GNode(id=frame["id"], name=frame.get("name", ""), role="screen", figma_type=frame["type"],
                   bbox=frame.get("absoluteBoundingBox"), layout=layout_of(frame),
                   children=reading_order(kids))
    infer_regions(screen)
    if screen.bbox:
        add_relative_boxes(screen, screen.bbox)
    return screen


def find_screens(n: dict) -> list[dict]:
    t = n.get("type")
    if not visible(n):
        return []
    if t in SCREEN_TYPES:
        return [n]
    if t in ("DOCUMENT", "CANVAS", "SECTION"):
        return [s for c in n.get("children", []) or [] for s in find_screens(c)]
    return []


def build_graph(payload: dict, max_depth: int = 12) -> dict:
    ctx = Ctx(payload.get("components", {}), payload.get("componentSets", {}), max_depth)
    screens = [make_screen(f, ctx) for f in find_screens(payload["document"])]
    return {
        "source": "figma",
        "file": payload.get("fileName"),
        "lastModified": payload.get("lastModified"),
        "screens": [s.to_dict() for s in screens],
    }


# --------------------------------------------------------------------------- output
def print_tree(d: dict, prefix: str = "", last: bool = True, root: bool = True) -> None:
    label = d["name"]
    tag = d["role"]
    if d.get("component") and d["component"] != d["name"]:
        tag += f": {d['component']}"
    extra = []
    if d.get("layout"):
        extra.append(d["layout"])
    if d.get("variant"):
        extra.append(",".join(f"{k}={v}" for k, v in d["variant"].items()))
    if d.get("text"):
        extra.append(repr(d["text"][:40]))
    if d.get("intent"):
        extra.append(f"→ {d['intent']}")
    line = f"{label}  [{tag}]" + (f"  {' | '.join(extra)}" if extra else "")
    print(line if root else f"{prefix}{'└── ' if last else '├── '}{line}")
    kids = d.get("children", [])
    for i, c in enumerate(kids):
        print_tree(c, prefix if root else prefix + ("    " if last else "│   "), i == len(kids) - 1, False)


def load_dotenv_file(path: str = ".env") -> None:
    """Load .env (python-dotenv if installed, else a minimal parser). Real env vars win."""
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        pass
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))


ROOT = Path(__file__).resolve().parent
DEFAULT_INTENT_FILE = "intents.json"


def resolve_intent_map(setting: str | None) -> str | None:
    """INTENT_MAP unset -> intents.json in the project root (next to this script), then CWD.
    INTENT_MAP=none disables it; any other value is a path (relative paths resolve from root)."""
    if setting and setting.lower() == "none":
        return None
    if setting:
        p = Path(setting)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"INTENT_MAP not found: {setting}")
        return str(p)
    for p in (ROOT / DEFAULT_INTENT_FILE, Path.cwd() / DEFAULT_INTENT_FILE):
        if p.exists():
            return str(p)
    return None


def env(name: str, default=None, cast=str):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    if cast is bool:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return cast(v.strip())


def main() -> None:
    """All configuration comes from .env (see .env.example). Only --env picks a different file."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env", help="path to the .env file (default: ./.env)")
    load_dotenv_file(ap.parse_args().env)

    raw = env("FIGMA_RAW")                                   # offline: build from a saved payload
    target = env("FIGMA_URL") or env("FIGMA_FILE_KEY")
    if raw:
        payload = json.loads(Path(raw).read_text())
        key, node_id = "raw", None
    else:
        if not target:
            sys.exit("Set FIGMA_URL or FIGMA_FILE_KEY in .env")
        token = env("FIGMA_TOKEN") or sys.exit("Set FIGMA_TOKEN in .env")
        key, node_id = parse_figma_url(target)
        payload = fetch_payload(key, node_id, token, env("FIGMA_REFRESH", False, bool))

    graph = build_graph(payload, env("MAX_DEPTH", 12, int))

    label = env("INTENT_LABEL", "rules")                      # rules | ollama | copilot | deepseek | mod:Class
    intent_map_path = resolve_intent_map(env("INTENT_MAP"))
    if intent_map_path:
        print(f"[intent] using intent map {intent_map_path}", file=sys.stderr)
    from intent import IntentMap, get_backend, label_graph, label_rules
    imap = IntentMap.load(intent_map_path) if intent_map_path else None
    if label in ("rules", "none"):
        if label == "rules":
            label_rules(graph, imap)
    else:
        import asyncio
        log = lambda m: print(m, file=sys.stderr)
        asyncio.run(label_graph(
            graph, get_backend(label, env("INTENT_MODEL")),
            chunk_size=env("INTENT_CHUNK_SIZE", 60, int),
            concurrency=env("INTENT_CONCURRENCY", 3, int),
            intent_map=imap,
            only_uncertain=not env("INTENT_LLM_ALL", False, bool),
            max_requests=env("INTENT_MAX_REQUESTS", None, int),
            log=log))

    from semantics import build_index
    index = build_index(graph, source="figma", meta={
        "file_key": key, "node_id": node_id, "file_name": payload.get("fileName"),
        "last_modified": payload.get("lastModified"),
        "labeller": label, "model": env("INTENT_MODEL"), "intent_map": intent_map_path,
    })

    out = Path(env("OUTPUT_PATH") or f"out/figma_{key}_{(node_id or 'root').replace(':', '-')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    if env("SAVE_TREE", False, bool):                        # optional nested view, for humans
        out.with_suffix(".tree.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False))

    if not env("QUIET", False, bool):
        for s in graph["screens"]:
            print_tree(s)
            print()
    print(f"wrote {out} ({len(index['screens'])} screen(s), {len(index['elements'])} elements)",
          file=sys.stderr)


if __name__ == "__main__":
    main()