def blocks_to_text(blocks) -> str:
    if isinstance(blocks, str): return blocks
    return " ".join(
        b.get("text", "") if isinstance(b, dict) else getattr(getattr(b, "text", None), "value", "")
        for b in blocks if b
    ).strip()
