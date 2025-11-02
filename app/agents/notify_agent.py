from __future__ import annotations
from typing import Any, Dict

def _percent(p: float | None) -> str:
    try:
        return f"{(p or 0.0)*100:.1f}%"
    except Exception:
        return "0%"

def format_weee_message_pt(result: Dict[str, Any]) -> str:
    """
    Formata uma mensagem em PT-BR com o resultado da classificação WEEE.
    """
    cat = (result.get("category") or {})
    cid, cname = cat.get("id"), cat.get("name")
    conf = result.get("confidence") or 0.0
    obj = result.get("top_object") or {}
    parts: list[str] = []

    filtered = result.get("filtered")
    if filtered and filtered.get("reason") == "non_eee":
        parts.append("Não parece ser um resíduo de EEE (WEEE).")
    else:
        if cid or cname:
            label = f"{cid} - {cname}" if cid and cname else (cname or str(cid))
            parts.append(f"Classificação WEEE: {label}")
        if obj.get("name"):
            parts.append(f"Objeto detectado: {obj['name']}")
        if obj.get("caption"):
            parts.append(f'Legenda do recorte: "{obj["caption"]}"')
        if result.get("image_caption"):
            parts.append(f'Legenda geral: "{result["image_caption"]}"')
        parts.append(f"Confiança visual: {_percent(conf)}")

    return "\n".join(parts)