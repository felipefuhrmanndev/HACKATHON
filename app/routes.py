from flask import render_template, request, jsonify, Response
from pathlib import Path
import logging
import os
from typing import Dict, Any, List, Tuple
import requests
import re  # <— adicionado
import json

from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
try:
    from twilio.request_validator import RequestValidator
except Exception:
    RequestValidator = None

from .services.vision import build_client, analyze_image_bytes
from .agents.weee_classifier import classify_image_bytes as classify_weee
from .agents.notify_agent import format_weee_message_pt

logger = logging.getLogger(__name__)

# ------------------------
# Heurísticas de deduplicação
# ------------------------
_PARENT_DEVICE_KEYWORDS = [
    "laptop","notebook","computador","pc","desktop","monitor","televisor","tv","tablet",
    "telefone","celular","smartphone","impressora","printer","roteador","router",
    "geladeira","refrigerador","frigorífico","freezer","forno","micro-ondas","lava-louças","máquina de lavar","secadora",
    "ar-condicionado","bomba de calor"
]
_SUBPART_KEYWORDS = [
    "keyboard","teclado","trackpad","touchpad","screen","display","panel","painel","bezel","stand",
    "mouse","speaker","alto-falante","alto falante","cable","cables","wire","fio","cabos","cabo"
]
_NON_EEE_KEYWORDS = [
    "pessoa","person","homem","mulher","dog","cachorro","cat","gato","animal",
    "árvore","tree","plant","planta","car","carro","bike","bicycle","food","comida","house","casa","building","prédio","rua","street"
]

def _norm(s: str | None) -> str:
    return (s or "").strip().lower()

def _hit(text: str, words: List[str]) -> bool:
    t = _norm(text)
    return any(w in t for w in words)

def _bbox_iou(a: Dict[str,int], b: Dict[str,int]) -> float:
    Ax, Ay, Aw, Ah = a["x"], a["y"], a["w"], a["h"]
    Bx, By, Bw, Bh = b["x"], b["y"], b["w"], b["h"]
    A_x2, A_y2 = Ax + Aw, Ay + Ah
    B_x2, B_y2 = Bx + Bw, By + Bh
    inter_x1 = max(Ax, Bx)
    inter_y1 = max(Ay, By)
    inter_x2 = min(A_x2, B_x2)
    inter_y2 = min(A_y2, B_y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = Aw * Ah
    areaB = Bw * Bh
    union = areaA + areaB - inter_area
    return (inter_area / union) if union > 0 else 0.0

def _public_url(path: str) -> str:
    base = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not base or path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"

# --- Normalização de números para Meta (remove 'whatsapp:' e não dígitos) ---
def _normalize_phone_for_meta(s: str | None) -> str:
    raw = (s or "").strip()
    raw = raw.replace("whatsapp:", "")
    digits = re.sub(r"\D", "", raw)
    return digits

# --- Envio via WhatsApp Cloud API (Meta) ---
def _send_whatsapp(to: str, body: str, media_url: str | None = None) -> None:
    """
    Envia mensagem via WhatsApp Cloud API.
    Usa META_WHATSAPP_TOKEN e META_WHATSAPP_PHONE_ID do .env.
    """
    token = os.getenv("META_WHATSAPP_TOKEN")
    phone_id = os.getenv("META_WHATSAPP_PHONE_ID")
    if not token or not phone_id:
        raise RuntimeError("META_WHATSAPP_TOKEN/META_WHATSAPP_PHONE_ID não configurados.")

    url = f"https://graph.facebook.com/v17.0/{phone_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    to_num = _normalize_phone_for_meta(to)
    payload: Dict[str, Any] = {"messaging_product": "whatsapp", "to": to_num}

    if media_url:
        payload.update({"type": "image", "image": {"link": _public_url(media_url), "caption": body}})
    else:
        payload.update({"type": "text", "text": {"body": body}})

    logger.info("Sending WhatsApp message via Meta to=%s media=%s", to_num, bool(media_url))
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"WhatsApp Cloud API error: {resp.status_code} {resp.text}")

# --- Download de mídia recebida (Meta) ---
def _meta_download_media_bytes(media_id: str) -> bytes:
    """
    1) GET /{media_id} -> retorna {url}
    2) GET url -> retorna bytes (com Authorization)
    """
    token = os.getenv("META_WHATSAPP_TOKEN")
    if not token:
        raise RuntimeError("META_WHATSAPP_TOKEN não configurado.")
    base = "https://graph.facebook.com/v17.0"
    # 1) Obter URL da mídia
    r1 = requests.get(f"{base}/{media_id}", headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r1.status_code != 200:
        raise RuntimeError(f"Falha ao resolver mídia {media_id}: {r1.status_code} {r1.text}")
    media_url = r1.json().get("url")
    if not media_url:
        raise RuntimeError("URL de mídia ausente na resposta da Graph API.")
    # 2) Baixar bytes
    r2 = requests.get(media_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    if r2.status_code != 200:
        raise RuntimeError(f"Falha ao baixar mídia: {r2.status_code}")
    return r2.content

def _twilio_client() -> TwilioClient:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        raise RuntimeError("TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN não configurados.")
    return TwilioClient(sid, token)

def _validate_twilio_signature(req) -> bool:
    if (os.getenv("TWILIO_VALIDATE_SIGNATURE", "false").lower() not in ("1","true","yes")):
        return True
    if RequestValidator is None:
        logger.warning("Twilio RequestValidator indisponível; sem validação.")
        return True
    token = os.getenv("TWILIO_AUTH_TOKEN")
    base = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not token or not base:
        logger.warning("AuthToken/PUBLIC_BASE_URL ausentes; sem validação.")
        return True
    validator = RequestValidator(token)
    signature = req.headers.get("X-Twilio-Signature", "")
    url = f"{base}{req.path}"
    return bool(validator.validate(url, req.form.to_dict(flat=True), signature))

def index():
    return render_template("index.html")

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

        # 1) Análise visual (gera crops e metadados)
        result = analyze_image_bytes(img_bytes, client, static_dir=static_dir)

        # 2) Pré-processamento para deduplicação e filtro de "não EEE"
        objs = result.get("objects", [])
        # Monta textos auxiliares
        for o in objs:
            o["_label_text"] = _norm(f"{o.get('name','')} {o.get('caption','')}")
            o["_is_parent"] = _hit(o["_label_text"], _PARENT_DEVICE_KEYWORDS)
            o["_is_subpart"] = _hit(o["_label_text"], _SUBPART_KEYWORDS)
            o["_is_non_eee"] = _hit(o["_label_text"], _NON_EEE_KEYWORDS)

        # Índices de pais
        parent_idxs = [i for i, o in enumerate(objs) if o.get("_is_parent")]
        # 3) Classificar cada recorte (com deduplicação)
        use_llm = os.getenv("USE_LLM_ARBITER", "false").lower() in ("1", "true", "yes")

        for i, obj in enumerate(objs):
            obj["weee_category_id"] = None
            obj["weee_category_name"] = None
            obj["weee_category"] = None

            # Ignorar se for claramente "não EEE"
            if obj.get("_is_non_eee") and not obj.get("_is_parent"):
                obj["weee_category"] = "Ignorado (não EEE)"
                continue

            # Se for subparte e houver um pai com sobreposição razoável, ignore para evitar duplicidade
            if obj.get("_is_subpart") and obj.get("bbox") and parent_idxs:
                for j in parent_idxs:
                    if i == j or not objs[j].get("bbox"):
                        continue
                    if _bbox_iou(obj["bbox"], objs[j]["bbox"]) >= 0.2:
                        obj["weee_category"] = f"Ignorado (parte de {objs[j].get('name','dispositivo')})"
                        break
                if obj["weee_category"]:
                    continue  # já ignorado por ser subparte do pai

            crop_url = obj.get("crop_url")
            if not crop_url:
                continue

            # Mapear "/static/..." -> app/static/...
            static_dir = Path(__file__).resolve().parents[0] / "static"
            rel = crop_url[len("/static/"):] if crop_url.startswith("/static/") else crop_url.lstrip("/")
            crop_path = static_dir / rel
            if not crop_path.exists():
                logger.warning("Crop não encontrado para classificação: %s", crop_path)
                continue

            try:
                crop_bytes = crop_path.read_bytes()
                cls = classify_weee(crop_bytes, client, static_dir=static_dir, use_llm_arbiter=use_llm)
                category = cls.get("category") or {}
                # Respeita filtro do classificador (não EEE)
                if cls.get("category") is None and cls.get("filtered", {}).get("reason") == "non_eee":
                    obj["weee_category"] = "Ignorado (não EEE)"
                    continue

                cid = category.get("id")
                cname = category.get("name")
                if cid or cname:
                    obj["weee_category_id"] = cid
                    obj["weee_category_name"] = cname
                    obj["weee_category"] = f"{cid} - {cname}" if cid and cname else (cname or str(cid))
            except Exception:
                logger.exception("Erro ao classificar crop individual")

        return render_template("results.html", result=result)
    except Exception as e:
        logger.exception("Erro ao analisar imagem")
        return f"Erro ao analisar imagem: {str(e)}\n\nVerifique os logs.", 500

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

# --- Webhook antigo do Twilio (mantido para compatibilidade; envio agora usa Meta) ---
def twilio_whatsapp():
    """
    Fluxo:
    - Se a mensagem vem do USER01 e contém imagem => classifica e envia laudo ao USER02.
    - Se a mensagem vem do USER02 => encaminha confirmação ao USER01.
    """
    try:
        logger.info("Incoming Twilio webhook: path=%s", request.path)
        if not _validate_twilio_signature(request):
            logger.warning("Twilio signature validation failed for request from %s", request.form.get("From"))
            return "Invalid signature", 403

        user01 = (os.getenv("USER01_WHATSAPP") or "").strip()
        user02 = (os.getenv("USER02_WHATSAPP") or "").strip()
        from_number = (request.form.get("From") or "").strip()
        body_text = (request.form.get("Body") or "").strip()
        num_media = int(request.form.get("NumMedia", "0") or "0")

        logger.info("Webhook from=%s num_media=%s user01=%s user02=%s", from_number, num_media, user01, user02)

        resp = MessagingResponse()
        msg = resp.message()

        if not user01 or not user02:
            msg.body("Configuração de usuários não encontrada no servidor.")
            return Response(str(resp), mimetype="application/xml")

        # Caso 1: Usuário 01 envia imagem
        if from_number == user01:
            if num_media <= 0:
                msg.body("Envie uma imagem para classificação WEEE.")
                return Response(str(resp), mimetype="application/xml")

            media_url = request.form.get("MediaUrl0")
            media_type = request.form.get("MediaContentType0") or ""
            if not media_url or not media_type.startswith("image/"):
                msg.body("A mensagem não contém imagem suportada. Envie JPG/PNG.")
                return Response(str(resp), mimetype="application/xml")

            sid = os.getenv("TWILIO_ACCOUNT_SID")
            token = os.getenv("TWILIO_AUTH_TOKEN")
            logger.info("Downloading media from %s", media_url)
            try:
                r = requests.get(media_url, auth=(sid, token), timeout=30)
            except Exception as ex:
                logger.exception("Erro ao baixar mídia: %s", ex)
                msg.body("Falha ao baixar a imagem. Tente novamente.")
                return Response(str(resp), mimetype="application/xml")

            logger.info("Media download response: %s", r.status_code)
            if r.status_code != 200:
                logger.warning("Falha ao baixar a imagem: status %s", r.status_code)
                msg.body("Falha ao baixar a imagem. Tente novamente.")
                return Response(str(resp), mimetype="application/xml")
            img_bytes = r.content

            # Classificar
            client = build_client()
            static_dir = Path(__file__).resolve().parents[0] / "static"
            use_llm = os.getenv("USE_LLM_ARBITER", "false").lower() in ("1","true","yes")
            logger.info("Calling classify_weee for incoming media (size=%s bytes) use_llm=%s", len(img_bytes), use_llm)
            result = classify_weee(img_bytes, client, static_dir=static_dir, use_llm_arbiter=use_llm)
            logger.info("Classification result keys: %s", ",".join(result.keys() if isinstance(result, dict) else []))

            # Preparar mensagem para o usuário 02
            text_pt = format_weee_message_pt(result)
            top_obj = result.get("top_object") or {}
            crop_url = top_obj.get("crop_url") or ""
            if crop_url:
                crop_url = _public_url(crop_url)
            # Envia ao usuário 02 (com mídia do crop, se houver)
            try:
                logger.info("Sending report to reviewer %s media=%s", user02, crop_url or None)
                _send_whatsapp(user02, text_pt, media_url=crop_url or None)
                logger.info("Report sent to reviewer %s", user02)
            except Exception as send_err:
                logger.exception("Erro ao enviar mensagem ao usuário 02: %s", send_err)
                msg.body("Erro ao encaminhar o laudo ao revisor.")
                return Response(str(resp), mimetype="application/xml")

            # Confirma ao usuário 01
            msg.body("Imagem recebida. O laudo foi enviado ao revisor (usuário 02).")
            return Response(str(resp), mimetype="application/xml")

        # Caso 2: Usuário 02 confirma
        if from_number == user02:
            # Encaminha confirmação ao usuário 01
            try:
                txt = body_text or "Confirmação recebida."
                _send_whatsapp(user01, f"Confirmação do revisor: {txt}")
            except Exception:
                logger.exception("Erro ao encaminhar confirmação ao usuário 01")
                msg.body("Erro ao enviar a confirmação ao solicitante.")
                return Response(str(resp), mimetype="application/xml")

            # Responde ao usuário 02
            msg.body("Confirmação recebida e enviada ao solicitante (usuário 01).")
            return Response(str(resp), mimetype="application/xml")

        # Outros remetentes
        msg.body("Este número aceita: (1) imagem de USER01 para classificação, (2) confirmação de USER02.")
        return Response(str(resp), mimetype="application/xml")

    except Exception:
        logger.exception("Erro no webhook do WhatsApp")
        resp = MessagingResponse()
        resp.message("Erro ao processar a mensagem. Tente novamente.")
        return Response(str(resp), mimetype="application/xml")

# --- Novo webhook da Meta (WhatsApp Cloud API) ---
def meta_whatsapp():
    """
    GET: verificação de webhook (hub.challenge)
    POST: eventos de mensagens
    - USER01 envia imagem -> classifica e envia laudo ao USER02
    - USER02 envia confirmação -> repassa ao USER01
    """
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == (os.getenv("META_WEBHOOK_VERIFY_TOKEN") or ""):
            return Response(challenge or "", status=200, mimetype="text/plain")
        return Response("forbidden", status=403)

    # POST
    try:
        data = request.get_json(silent=True) or {}
        logger.info("Incoming Meta webhook: keys=%s", list(data.keys()))
        entries = data.get("entry", [])
        if not entries:
            return jsonify(status="ignored"), 200

        user01 = (os.getenv("USER01_WHATSAPP") or "").strip()
        user02 = (os.getenv("USER02_WHATSAPP") or "").strip()
        n_user01 = _normalize_phone_for_meta(user01)
        n_user02 = _normalize_phone_for_meta(user02)

        for entry in entries:
            changes = entry.get("changes", [])
            for ch in changes:
                value = ch.get("value", {})
                messages = value.get("messages", []) or []
                contacts = value.get("contacts", []) or []
                for msg in messages:
                    from_raw = msg.get("from", "")
                    n_from = _normalize_phone_for_meta(from_raw)
                    mtype = msg.get("type")
                    body_text = ""
                    img_bytes = None

                    if mtype == "text":
                        body_text = (msg.get("text", {}) or {}).get("body", "") or ""
                    elif mtype == "image":
                        media_id = (msg.get("image", {}) or {}).get("id")
                        if media_id:
                            try:
                                img_bytes = _meta_download_media_bytes(media_id)
                            except Exception:
                                logger.exception("Falha ao baixar imagem (Meta)")
                                # Segue para próxima mensagem
                                continue
                    else:
                        logger.info("Tipo de mensagem não suportado: %s", mtype)
                        continue

                    # Caso 1: USER01 envia imagem para classificar
                    if n_from == n_user01 and img_bytes:
                        try:
                            client = build_client()
                            static_dir = Path(__file__).resolve().parents[0] / "static"
                            use_llm = os.getenv("USE_LLM_ARBITER", "false").lower() in ("1","true","yes")
                            result = classify_weee(img_bytes, client, static_dir=static_dir, use_llm_arbiter=use_llm)
                            text_pt = format_weee_message_pt(result)
                            top_obj = result.get("top_object") or {}
                            crop_url = top_obj.get("crop_url") or ""
                            # Envia ao USER02 (revisor) com crop
                            _send_whatsapp(user02, text_pt, media_url=crop_url or None)
                            # Confirma ao USER01
                            _send_whatsapp(user01, "Imagem recebida. O laudo foi enviado ao revisor (usuário 02).")
                        except Exception:
                            logger.exception("Erro ao processar/classificar mensagem de USER01")
                        continue

                    # Caso 2: USER02 envia confirmação (texto)
                    if n_from == n_user02 and body_text:
                        try:
                            _send_whatsapp(user01, f"Confirmação do revisor: {body_text}")
                            _send_whatsapp(user02, "Confirmação recebida e enviada ao solicitante (usuário 01).")
                        except Exception:
                            logger.exception("Erro ao encaminhar confirmação USER02 -> USER01")
                        continue

        return jsonify(status="ok"), 200
    except Exception:
        logger.exception("Erro no webhook da Meta")
        return jsonify(error="internal_error"), 200  # 200 para não reenfileirar excessivamente

# --- Telegram helpers ---
def _tg_api(method: str) -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN não configurado.")
    return f"https://api.telegram.org/bot{token}/{method}"

def _tg_file_url(file_path: str) -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    return f"https://api.telegram.org/file/bot{token}/{file_path}"

def _tg_send_text(chat_id: str | int, text: str) -> None:
    url = _tg_api("sendMessage")
    r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=30)
    if r.status_code >= 400:
        logger.error("Telegram sendMessage error: %s %s", r.status_code, r.text)

def _tg_send_photo(chat_id: str | int, photo_url: str, caption: str | None = None) -> None:
    url = _tg_api("sendPhoto")
    payload = {"chat_id": chat_id, "photo": photo_url}
    if caption:
        payload["caption"] = caption
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code >= 400:
        logger.error("Telegram sendPhoto error: %s %s", r.status_code, r.text)

def telegram_webhook():
    """
    Recebe updates do Telegram:
    - USER01 envia foto -> classifica -> envia laudo + crop a USER02
    - USER02 envia texto -> repassa confirmação a USER01
    """
    try:
        data = request.get_json(silent=True) or {}
        logger.info("Incoming Telegram webhook: keys=%s", list(data.keys()))
        msg = (data.get("message") or {}) if "message" in data else (data.get("edited_message") or {})
        if not msg:
            return jsonify(ok=True)

        user01 = os.getenv("TELEGRAM_USER01_ID")
        user02 = os.getenv("TELEGRAM_USER02_ID")
        chat_id = str(msg.get("chat", {}).get("id") or "")
        text = msg.get("text", "") or ""
        photos = msg.get("photo", []) or []

        logger.info("Telegram msg chat_id=%s has_text=%s has_photos=%s", chat_id, bool(text), bool(photos))

        # Caso 1: USER01 envia uma foto
        if user01 and chat_id == str(user01) and photos:
            # pega a melhor resolução (última)
            file_id = photos[-1].get("file_id")
            # 1) getFile
            r = requests.get(_tg_api("getFile"), params={"file_id": file_id}, timeout=30)
            if r.status_code != 200:
                logger.error("Telegram getFile falhou: %s %s", r.status_code, r.text)
                _tg_send_text(chat_id, "Falha ao baixar a imagem.")
                return jsonify(ok=True)
            file_path = (r.json().get("result") or {}).get("file_path")
            if not file_path:
                _tg_send_text(chat_id, "Arquivo não encontrado.")
                return jsonify(ok=True)
            # 2) baixar bytes
            img_url = _tg_file_url(file_path)
            rb = requests.get(img_url, timeout=60)
            if rb.status_code != 200:
                _tg_send_text(chat_id, "Falha ao baixar a imagem do Telegram.")
                return jsonify(ok=True)
            img_bytes = rb.content

            # Classificar
            client = build_client()
            static_dir = Path(__file__).resolve().parents[0] / "static"
            use_llm = os.getenv("USE_LLM_ARBITER", "false").lower() in ("1","true","yes")
            logger.info("Classifying Telegram image (bytes=%s) use_llm=%s", len(img_bytes), use_llm)
            result = classify_weee(img_bytes, client, static_dir=static_dir, use_llm_arbiter=use_llm)
            text_pt = format_weee_message_pt(result)
            top_obj = result.get("top_object") or {}
            crop_url = top_obj.get("crop_url") or ""

            # Envia ao USER02
            if user02:
                if crop_url:
                    logger.info("Sending photo to reviewer chat_id=%s", user02)
                    _tg_send_photo(user02, _public_url(crop_url), caption=text_pt)
                else:
                    logger.info("Sending text to reviewer chat_id=%s", user02)
                    _tg_send_text(user02, text_pt)
            # Confirma ao USER01
            logger.info("Confirming to requester chat_id=%s", user01)
            _tg_send_text(user01, "Imagem recebida. O laudo foi enviado ao revisor (usuário 02).")
            return jsonify(ok=True)

        # Caso 2: USER02 envia confirmação (texto)
        if user02 and chat_id == str(user02) and text:
            if user01:
                logger.info("Forwarding confirmation from reviewer %s to requester %s", user02, user01)
                _tg_send_text(user01, f"Confirmação do revisor: {text}")
            _tg_send_text(user02, "Confirmação recebida e enviada ao solicitante (usuário 01).")
            return jsonify(ok=True)

        # Outros casos: ignore educadamente
        logger.info("Unhandled Telegram chat_id=%s; sending helper message.", chat_id)
        _tg_send_text(chat_id, "Envie uma foto (USER01) ou um texto de confirmação (USER02).")
        return jsonify(ok=True)

    except Exception:
        logger.exception("Erro no webhook do Telegram")
        return jsonify(ok=True)

def register_routes(app):
    """Registra rotas diretamente no app, sem Blueprint."""
    app.add_url_rule("/", view_func=index, methods=["GET"])
    app.add_url_rule("/analyze", view_func=analyze, methods=["POST"])
    app.add_url_rule("/api/analyze", view_func=api_analyze, methods=["POST"])
    app.add_url_rule("/api/classify", view_func=api_classify, methods=["POST"])
    # Webhooks
    app.add_url_rule("/twilio/whatsapp", view_func=twilio_whatsapp, methods=["POST"])
    app.add_url_rule("/meta/whatsapp", view_func=meta_whatsapp, methods=["GET", "POST"])
    app.add_url_rule("/telegram/webhook", view_func=telegram_webhook, methods=["POST"])