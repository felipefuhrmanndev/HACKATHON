import io
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple

from PIL import Image
from dotenv import load_dotenv

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def build_client(endpoint: str | None = None, key: str | None = None) -> ImageAnalysisClient:
    load_dotenv()
    endpoint = endpoint or os.getenv("AI_SERVICE_ENDPOINT")
    key = key or os.getenv("AI_SERVICE_KEY")
    if not endpoint or not key:
        raise RuntimeError("AI_SERVICE_ENDPOINT/AI_SERVICE_KEY não configurados (.env).")
    return ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _save_image_bytes(image_bytes: bytes, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    with open(out_path, "wb") as f:
        f.write(image_bytes)

def _image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def _resize_to_valid(img: Image.Image, min_side: int = 50, max_side: int = 16000) -> Image.Image:
    w, h = img.size
    scale = 1.0
    if min(w, h) < min_side:
        scale = max(scale, min_side / float(min(w, h)))
    if max(w, h) > max_side:
        scale = min(scale, max_side / float(max(w, h))) if scale == 1.0 else min(scale, max_side / float(max(w, h)))
    if scale != 1.0:
        new_w = max(min_side, min(int(round(w * scale)), max_side))
        new_h = max(min_side, min(int(round(h * scale)), max_side))
        return img.resize((new_w, new_h), Image.BICUBIC)
    return img

def _crop_regions(img: Image.Image, bboxes: List[Dict[str, int]]) -> List[Image.Image]:
    crops: List[Image.Image] = []
    W, H = img.size
    for r in bboxes:
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        x2, y2 = max(0, x), max(0, y)
        x3, y3 = min(W, x + w), min(H, y + h)
        if x3 > x2 and y3 > y2:
            crops.append(img.crop((x2, y2, x3, y3)))
    return crops

def _iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
    # boxes: (x, y, w, h)
    Ax, Ay, Aw, Ah = boxA
    Bx, By, Bw, Bh = boxB
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
    if union <= 0:
        return 0.0
    return inter_area / union

def analyze_image_bytes(
    image_bytes: bytes,
    client: ImageAnalysisClient,
    static_dir: Path | None = None,
    enable_grid_fallback: bool = True,
    min_side: int = 50,
    max_side: int = 16000
) -> Dict[str, Any]:
    # Diretórios de saída (estáticos para o Flask)
    base_static = static_dir or Path(__file__).resolve().parents[1] / "static"
    upload_dir = base_static / "uploads"
    crops_root = base_static / "crops"
    session_id = uuid.uuid4().hex
    session_upload_dir = upload_dir / session_id
    session_crops_dir = crops_root / session_id
    _ensure_dir(session_upload_dir)
    _ensure_dir(session_crops_dir)

    # Abrir imagem e ajustar ao intervalo suportado pelo serviço (50..16000)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img = _resize_to_valid(pil_img, min_side=min_side, max_side=max_side)
    W, H = pil_img.size

    # Salvar "original" (versão válida para o serviço)
    original_path = session_upload_dir / "original.jpg"
    _save_image_bytes(_image_to_bytes(pil_img), original_path)

    # Rodar análise para legenda geral + objetos
    analysis = client.analyze(
        image_data=_image_to_bytes(pil_img),
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.OBJECTS],
    )

    # Extrair legenda geral
    image_caption = analysis.caption.text if getattr(analysis, "caption", None) else None

    # Extrair objetos
    objects: List[Dict[str, Any]] = []
    bboxes: List[Dict[str, int]] = []

    detected_objects = []
    if getattr(analysis, "objects", None) is not None:
        detected_objects = getattr(analysis.objects, "list", analysis.objects) or []

    for det in detected_objects:
        r = det.bounding_box
        name = getattr(det, "name", None)
        confidence = getattr(det, "confidence", None)
        if (name is None or confidence is None) and hasattr(det, "tags") and det.tags:
            tag0 = det.tags[0]
            name = name or getattr(tag0, "name", None)
            confidence = confidence if confidence is not None else getattr(tag0, "confidence", None)
        name = name or "objeto"
        confidence = float(confidence) if confidence is not None else 0.0

        bbox = {"x": int(r.x), "y": int(r.y), "w": int(r.width), "h": int(r.height)}
        bboxes.append(bbox)
        objects.append({"name": name, "confidence": confidence, "bbox": bbox})

    # Cortar objetos detectados e obter legenda por recorte
    crops = _crop_regions(pil_img, bboxes)

    for idx in range(len(objects)):
        obj = objects[idx]
        if idx >= len(crops):
            continue

        crop_img = crops[idx]

        # Salvar a imagem do recorte para exibição (sem forçar tamanho)
        crop_file = session_crops_dir / f"obj_{idx:03d}.jpg"
        _save_image_bytes(_image_to_bytes(crop_img), crop_file)

        # Para análise de legenda, garantir tamanho válido
        crop_for_vision = _resize_to_valid(crop_img, min_side=min_side, max_side=max_side)
        crop_analysis = client.analyze(
            image_data=_image_to_bytes(crop_for_vision),
            visual_features=[VisualFeatures.CAPTION],
        )
        obj_caption = crop_analysis.caption.text if getattr(crop_analysis, "caption", None) else None

        obj["crop_url"] = f"/static/crops/{session_id}/{crop_file.name}"
        obj["caption"] = obj_caption

    # Fallback por grade (opcional) — pula se desabilitado
    if enable_grid_fallback and len(objects) < 4:
        min_desired = 4
        max_extra = 10
        added = 0
        seen_captions = set([o.get("caption") for o in objects if o.get("caption")])

        grid_n = 3
        cell_w = W // grid_n
        cell_h = H // grid_n

        for gy in range(grid_n):
            for gx in range(grid_n):
                if added >= max_extra:
                    break
                cell_x = gx * cell_w
                cell_y = gy * cell_h
                cell_w_eff = cell_w if gx < grid_n - 1 else W - cell_x
                cell_h_eff = cell_h if gy < grid_n - 1 else H - cell_y

                # Evitar células muito pequenas (< 50px) para análise
                if cell_w_eff < min_side or cell_h_eff < min_side:
                    continue

                cell_box = (cell_x, cell_y, cell_w_eff, cell_h_eff)

                overlaps = False
                for b in bboxes:
                    if _iou(cell_box, (b["x"], b["y"], b["w"], b["h"])) > 0.2:
                        overlaps = True
                        break
                if overlaps:
                    continue

                extra_crop = pil_img.crop((cell_x, cell_y, cell_x + cell_w_eff, cell_y + cell_h_eff))
                extra_file = session_crops_dir / f"extra_{added:03d}.jpg"
                _save_image_bytes(_image_to_bytes(extra_crop), extra_file)

                # Garantir tamanho válido também aqui
                extra_for_vision = _resize_to_valid(extra_crop, min_side=min_side, max_side=max_side)
                extra_analysis = client.analyze(
                    image_data=_image_to_bytes(extra_for_vision),
                    visual_features=[VisualFeatures.CAPTION],
                )
                extra_caption = extra_analysis.caption.text if getattr(extra_analysis, "caption", None) else None
                if not extra_caption or extra_caption.strip() == "":
                    continue
                if extra_caption in seen_captions:
                    continue

                extra_bbox = {"x": cell_x, "y": cell_y, "w": cell_w_eff, "h": cell_h_eff}
                objects.append({
                    "name": extra_caption,
                    "confidence": 0.0,
                    "bbox": extra_bbox,
                    "crop_url": f"/static/crops/{session_id}/{extra_file.name}",
                    "caption": extra_caption
                })
                bboxes.append(extra_bbox)
                seen_captions.add(extra_caption)
                added += 1
            if added >= max_extra:
                break

    return {
        "session_id": session_id,
        "original_url": f"/static/uploads/{session_id}/original.jpg",
        "image_caption": image_caption,
        "objects": objects
    }