from __future__ import annotations
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image

from app.services.vision import analyze_image_bytes as visual_analyze

# Categorias WEEE
CATEGORIES = [
    {"id": 1, "name": "Equipamentos de troca de temperatura"},
    {"id": 2, "name": "Ecrãs/Monitores (>100 cm²)"},
    {"id": 3, "name": "Lâmpadas"},
    {"id": 4, "name": "Grandes (dimensão > 50cm)"},
    {"id": 5, "name": "Pequenos (dimensão <= 50cm)"},
    {"id": 6, "name": "TIC pequena dimensão (<= 50cm)"},
]

# Palavras-chave por categoria (PT/EN)
KEYWORDS = {
    1: ["refrigerador","geladeira","frigorífico","freezer","congelador","ar-condicionado","ar condicionado","condicionador de ar","bomba de calor","heat pump","air conditioner","fridge","cooler"],
    2: ["televisor","tv","smart tv","monitor","ecrã","tela","laptop","notebook","tablet","display","screen"],
    3: ["lâmpada","lampada","lamp","bulbo","light bulb","fluorescente","led","incandescente","tube","tubo"],
    4: ["máquina de lavar","lavadora","secadora","lava-louças","lava louças","dishwasher","stove","fogão","oven","forno","range","geladeira grande"],
    5: ["aspirador","micro-ondas","microondas","torradeira","ferro de passar","kettle","chaleira","liquidificador","blender","mixer","câmera","camera"],
    6: ["celular","telefone","smartphone","phone","pc pequeno","mini pc","router","roteador","gps","calculadora","calculator","modem","printer","impressora","tablet","notebook","laptop"],
}

# Itens claramente fora de EEE (pessoas, animais, veículos, natureza, alimentos, etc.)
NON_EEE_KEYWORDS = [
    "pessoa","person","homem","mulher","people","boy","girl",
    "dog","cachorro","cat","gato","animal","bird","passaro","cavalo",
    "árvore","tree","plant","planta","flor","flower","grass","grama","landscape","paisagem","sky","céu","beach","praia","ocean","mar","montanha",
    "car","carro","bike","bicycle","moto","motorcycle","truck","caminhão","bus","ônibus",
    "food","comida","fruta","fruit","vegetal","vegetable","drink","bebida",
    "house","casa","building","prédio","rua","street","wall","parede","sofa","couch","table","mesa","chair","cadeira","book","livro"
]

# Subpartes comuns que podem gerar duplicidade (partes de um dispositivo maior)
SUBPART_KEYWORDS = [
    "keyboard","teclado","trackpad","touchpad","bezel","cover","tampa",
    "screen","display","panel","painel","monitor stand","stand",
    "cable","cables","wire","fio","cabos","cabo","mouse","speaker","alto-falante","alto falante"
]

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _token_hits(text: str, keywords: List[str]) -> int:
    t = _norm(text)
    return sum(1 for k in keywords if k in t)

def _is_non_eee(text: str, cat_scores: Dict[int, int]) -> bool:
    # Se tiver qualquer indício de EEE, não filtra. Senão, exige >=2 matches de não-EEE.
    if sum(cat_scores.values()) > 0:
        return False
    return _token_hits(text, NON_EEE_KEYWORDS) >= 2

def _pick_best_object(objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not objects:
        return None
    best = objects[0]
    for o in objects[1:]:
        if o.get("confidence", 0.0) > best.get("confidence", 0.0):
            best = o
        elif o.get("confidence", 0.0) == best.get("confidence", 0.0):
            if o.get("caption") and not best.get("caption"):
                best = o
    return best

def _estimate_size_ratio(img_w: int, img_h: int, bbox: Dict[str, int]) -> float:
    iw, ih = max(1, img_w), max(1, img_h)
    bw, bh = max(1, bbox["w"]), max(1, bbox["h"])
    return (bw * bh) / (iw * ih)

def _size_to_bucket(size_ratio: float) -> str:
    return "grande" if size_ratio >= 0.20 else "pequeno"

@dataclass
class VisualAgentOutput:
    image_caption: Optional[str]
    objects: List[Dict[str, Any]]
    original_url: str
    session_id: str

def visual_agent(image_bytes: bytes, client, static_dir: Optional[Path]) -> VisualAgentOutput:
    # Desabilita o grid fallback ao classificar recortes (evita subcrops < 50px)
    vres = visual_analyze(image_bytes, client, static_dir=static_dir, enable_grid_fallback=False)
    return VisualAgentOutput(
        image_caption=vres.get("image_caption"),
        objects=vres.get("objects", []),
        original_url=vres.get("original_url", ""),
        session_id=vres.get("session_id", "")
    )

@dataclass
class SizeAgentOutput:
    img_size: Tuple[int, int]
    object_bbox: Optional[Dict[str, int]]
    size_ratio: Optional[float]
    size_bucket: Optional[str]

def size_agent(image_bytes: bytes, top_object: Optional[Dict[str, Any]]) -> SizeAgentOutput:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    if not top_object or not top_object.get("bbox"):
        return SizeAgentOutput((W, H), None, None, None)
    ratio = _estimate_size_ratio(W, H, top_object["bbox"])
    bucket = _size_to_bucket(ratio)
    return SizeAgentOutput((W, H), top_object["bbox"], ratio, bucket)

@dataclass
class RuleAgentOutput:
    category_id: int
    category_name: str
    score: int
    evidence: Dict[str, Any]

def rule_agent(image_caption: Optional[str], top_object: Optional[Dict[str, Any]], size_bucket: Optional[str]) -> RuleAgentOutput:
    texts = []
    if top_object:
        if top_object.get("name"): texts.append(str(top_object["name"]))
        if top_object.get("caption"): texts.append(str(top_object["caption"]))
    if image_caption:
        texts.append(str(image_caption))
    combined = " | ".join(texts)

    cat_scores = {cid: _token_hits(combined, kws) for cid, kws in KEYWORDS.items()}
    if all(s == 0 for s in cat_scores.values()):
        cid = 4 if size_bucket == "grande" else 5
        return RuleAgentOutput(
            category_id=cid,
            category_name=next(c["name"] for c in CATEGORIES if c["id"] == cid),
            score=1,
            evidence={"source": "size_fallback", "size_bucket": size_bucket, "text": combined}
        )

    pref = [1, 2, 3, 6, 4, 5]
    best_score = max(cat_scores.values())
    candidates = [cid for cid, sc in cat_scores.items() if sc == best_score]
    cid = sorted(candidates, key=lambda x: pref.index(x))[0]
    return RuleAgentOutput(
        category_id=cid,
        category_name=next(c["name"] for c in CATEGORIES if c["id"] == cid),
        score=best_score,
        evidence={"source": "keywords", "scores": cat_scores, "text": combined, "size_bucket": size_bucket}
    )

def _try_llm_arbiter(rule_choice: RuleAgentOutput, all_options: List[Dict[str, Any]], context: Dict[str, Any]) -> Optional[RuleAgentOutput]:
    try:
        endpoint = os.getenv("AGENTS_PROJECT_ENDPOINT")
        if not endpoint:
            return None
        model = os.getenv("AGENTS_MODEL_DEPLOYMENT", "gpt-4o")
        from azure.ai.agents import AgentsClient
        from azure.identity import DefaultAzureCredential

        client = AgentsClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True
            ),
        )
        with client:
            cats = ", ".join([f"{opt['id']}-{opt['name']}" for opt in all_options])
            instructions = (
                "Você é um árbitro que classifica resíduos de EEE em UMA das 6 categorias WEEE.\n"
                "Retorne somente:\n"
                "- id (1..6)\n"
                "- nome\n"
                "- justificativa breve\n\n"
                f"Categorias: {cats}\n"
                f"Sugestão de regras: {rule_choice.category_id}-{rule_choice.category_name}\n"
                f"Contexto: {context}\n"
            )
            agent = client.create_agent(model=model, name="weee_arbiter", instructions=instructions)
            thread = client.threads.create()
            client.messages.create(thread_id=thread.id, role="user", content="Escolha a melhor categoria.")
            run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
            if run.status == "failed":
                return None
            msgs = client.messages.list(thread_id=thread.id)
            text = ""
            for m in msgs:
                if m.text_messages:
                    text = m.text_messages[-1].text.value or ""
            chosen_id = None
            for c in all_options:
                if f"{c['id']}" in text or c["name"].lower() in text.lower():
                    chosen_id = c["id"]
                    break
            if not chosen_id:
                return None
            return RuleAgentOutput(
                category_id=chosen_id,
                category_name=next(c["name"] for c in CATEGORIES if c["id"] == chosen_id),
                score=rule_choice.score,
                evidence={"source": "llm_arbiter", "llm_text": text}
            )
    except Exception:
        return None

def classify_image_bytes(
    image_bytes: bytes,
    client,
    static_dir: Optional[Path] = None,
    use_llm_arbiter: bool = False
) -> Dict[str, Any]:
    vis = visual_agent(image_bytes, client, static_dir)
    top_obj = _pick_best_object(vis.objects)
    sz = size_agent(image_bytes, top_object=top_obj)

    # Filtro não-EEE (rápido)
    texts = []
    if top_obj:
        if top_obj.get("name"): texts.append(str(top_obj["name"]))
        if top_obj.get("caption"): texts.append(str(top_obj["caption"]))
    if vis.image_caption:
        texts.append(str(vis.image_caption))
    combined = " | ".join(texts)
    cat_scores = {cid: _token_hits(combined, kws) for cid, kws in KEYWORDS.items()}
    if _is_non_eee(combined, cat_scores):
        return {
            "session_id": vis.session_id,
            "original_url": vis.original_url,
            "image_caption": vis.image_caption,
            "top_object": top_obj,
            "agents": {
                "visual": {"objects": vis.objects},
                "size": {
                    "image_w": sz.img_size[0],
                    "image_h": sz.img_size[1],
                    "object_bbox": sz.object_bbox,
                    "size_ratio": sz.size_ratio,
                    "size_bucket": sz.size_bucket
                },
                "rules": None
            },
            "category": None,
            "confidence": float(top_obj.get("confidence", 0.0)) if top_obj else 0.0,
            "filtered": {"reason": "non_eee", "text": combined}
        }

    # Regras + árbitro opcional
    rules_choice = rule_agent(vis.image_caption, top_obj, sz.size_bucket)
    final_choice = rules_choice
    if use_llm_arbiter:
        arb = _try_llm_arbiter(
            rules_choice,
            CATEGORIES,
            context={
                "image_caption": vis.image_caption,
                "top_object": top_obj,
                "size_bucket": sz.size_bucket,
                "size_ratio": sz.size_ratio,
            }
        )
        if arb is not None:
            final_choice = arb

    return {
        "session_id": vis.session_id,
        "original_url": vis.original_url,
        "image_caption": vis.image_caption,
        "top_object": top_obj,
        "agents": {
            "visual": {"objects": vis.objects},
            "size": {
                "image_w": sz.img_size[0],
                "image_h": sz.img_size[1],
                "object_bbox": sz.object_bbox,
                "size_ratio": sz.size_ratio,
                "size_bucket": sz.size_bucket
            },
            "rules": {
                "category_id": rules_choice.category_id,
                "category_name": rules_choice.category_name,
                "score": rules_choice.score,
                "evidence": rules_choice.evidence
            },
        },
        "category": {"id": final_choice.category_id, "name": final_choice.category_name},
        "confidence": float(top_obj.get("confidence", 0.0)) if top_obj else 0.0,
        "explanation": final_choice.evidence if isinstance(final_choice.evidence, dict) else {"source": "rules"}
    }

from flask import Blueprint, render_template, request, jsonify
from pathlib import Path
import logging
import traceback
import os

from app.services.vision import build_client, analyze_image_bytes
# use local classifier function defined above instead of importing the same module
classify_weee = classify_image_bytes

bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

@bp.get("/")
def index():
    return render_template("index.html")

@bp.post("/analyze")
def analyze():
    if "image" not in request.files:
        return "Arquivo não enviado.", 400
    file = request.files["image"]
    if not file or file.filename == "":
        return "Arquivo inválido.", 400

    img_bytes = file.read()
    try:
        client = build_client()
        static_dir = Path(__file__).resolve().parents[0] / "static"

        # Chamada visual principal (retorna crops e metadados)
        result = analyze_image_bytes(img_bytes, client, static_dir=static_dir)

        # Para cada crop detectado, classificar apenas aquele recorte e anexar categoria WEEE
        for obj in result.get("objects", []):
            obj["weee_category"] = None
            try:
                crop_url = obj.get("crop_url")
                if not crop_url:
                    continue
                # crop_url: "/static/crops/{session_id}/{file.jpg}"
                rel = crop_url.lstrip("/static/")  # "crops/..."
                crop_path = static_dir / rel
                if not crop_path.exists():
                    continue
                crop_bytes = crop_path.read_bytes()
                # Classifica o recorte; desliga LLM arbiter por padrão (mais rápido)
                cls = classify_weee(crop_bytes, client, static_dir=static_dir, use_llm_arbiter=False)
                category = cls.get("category") or {}
                if category:
                    cid = category.get("id")
                    cname = category.get("name")
                    obj["weee_category"] = f"{cid} - {cname}" if cid and cname else (cname or None)
            except Exception:
                logger.exception("Erro ao classificar crop individual; seguindo sem categoria para este crop.")

        return render_template("results.html", result=result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Erro ao analisar imagem")
        return f"Erro ao analisar imagem: {str(e)}", 500

@bp.post("/api/analyze")
def api_analyze():
    if "image" not in request.files:
        return jsonify(error="Arquivo não enviado."), 400
    img_bytes = request.files["image"].read()
    try:
        client = build_client()
        static_dir = Path(__file__).resolve().parents[0] / "static"
        result = analyze_image_bytes(img_bytes, client, static_dir=static_dir)
        return jsonify(result)
    except Exception as e:
        logger.exception("Erro na API /api/analyze")
        return jsonify(error=str(e)), 500

@bp.post("/api/classify")
def api_classify():
    if "image" not in request.files:
        return jsonify(error="Arquivo não enviado."), 400
    img_bytes = request.files["image"].read()

    qs_llm = request.args.get("llm")
    if qs_llm is None:
        use_llm = os.getenv("USE_LLM_ARBITER", "false").lower() in ("1", "true", "yes")
    else:
        use_llm = qs_llm.lower() in ("1", "true", "yes")

    try:
        client = build_client()
        static_dir = Path(__file__).resolve().parents[0] / "static"
        result = classify_weee(img_bytes, client, static_dir=static_dir, use_llm_arbiter=use_llm)
        return jsonify(result)
    except Exception as e:
        logger.exception("Erro na API /api/classify")
        return jsonify(error=str(e)), 500