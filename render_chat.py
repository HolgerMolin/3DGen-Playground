#!/usr/bin/env python3
"""Render a Claude Code JSONL session transcript into a readable .txt chat log."""
import json, sys, textwrap

def trunc(s, n):
    s = s if isinstance(s, str) else str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"\n        ... [truncated, {len(s)-n} more chars]"

def render(path, out):
    lines = []
    for raw in open(path):
        try:
            o = json.loads(raw)
        except Exception:
            continue
        t = o.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = o.get("message", {}) or {}
        content = msg.get("content")

        if t == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    txt = (b.get("text") or "").strip()
                    # skip injected harness reminders / command wrappers
                    if not txt or txt.startswith("<"):
                        continue
                    lines.append("="*80)
                    lines.append("USER:")
                    lines.append(txt)
                    lines.append("")
                elif bt == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    c = (c or "").strip()
                    if c:
                        lines.append("    [tool result]")
                        lines.append(textwrap.indent(trunc(c, 700), "      "))
                        lines.append("")

        elif t == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    txt = (b.get("text") or "").strip()
                    if not txt:
                        continue
                    lines.append("-"*80)
                    lines.append("ASSISTANT:")
                    lines.append(txt)
                    lines.append("")
                elif bt == "tool_use":
                    name = b.get("name")
                    inp = b.get("input", {}) or {}
                    fp = inp.get("file_path", "")
                    header = f"    >> {name}" + (f"  [{fp}]" if fp else "")
                    lines.append(header)
                    if name in ("Edit",):
                        lines.append("       --- old ---")
                        lines.append(textwrap.indent(trunc(inp.get("old_string", ""), 1200), "       "))
                        lines.append("       --- new ---")
                        lines.append(textwrap.indent(trunc(inp.get("new_string", ""), 1200), "       "))
                    elif name in ("MultiEdit",):
                        for e in inp.get("edits", []):
                            lines.append("       --- old ---")
                            lines.append(textwrap.indent(trunc(e.get("old_string", ""), 800), "       "))
                            lines.append("       --- new ---")
                            lines.append(textwrap.indent(trunc(e.get("new_string", ""), 800), "       "))
                    elif name in ("Write",):
                        lines.append(textwrap.indent(trunc(inp.get("content", ""), 1500), "       "))
                    elif name in ("Bash",):
                        lines.append("       $ " + trunc(inp.get("command", ""), 400))
                    else:
                        # generic: show a compact view of inputs
                        compact = {k: trunc(str(v), 300) for k, v in inp.items() if k != "file_path"}
                        if compact:
                            lines.append(textwrap.indent(trunc(json.dumps(compact, indent=2), 600), "       "))
                    lines.append("")

    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    return len(lines)

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = render(src, dst)
    print(f"wrote {dst} ({n} lines)")
