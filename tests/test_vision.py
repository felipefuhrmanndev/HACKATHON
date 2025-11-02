import io
from pathlib import Path
from PIL import Image
import types

from app.services.vision import analyze_image_bytes
from azure.ai.vision.imageanalysis.models import VisualFeatures

class DummyBBox:
    def __init__(self, x, y, w, h): self.x, self.y, self.width, self.height = x, y, w, h

class DummyTag:
    def __init__(self, name, conf): self.name, self.confidence = name, conf

class DummyObj:
    def __init__(self): 
        self.bounding_box = DummyBBox(10, 10, 50, 50)
        self.tags = [DummyTag("car", 0.88)]

class DummyList:
    def __init__(self, items): self.list = items

class DummyCaption:
    def __init__(self, text): self.text, self.confidence = text, 0.95

class DummyAnalysis:
    def __init__(self, mode):
        if mode == "full":
            self.caption = DummyCaption("street with cars")
            self.objects = DummyList([DummyObj()])
        else:
            self.caption = DummyCaption("a car")

class DummyClient:
    def analyze(self, image_data=None, visual_features=None):
        if VisualFeatures.OBJECTS in visual_features:
            return DummyAnalysis("full")
        return DummyAnalysis("crop")

def _make_image_bytes(w=100, h=100, color=(255, 255, 255)):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_analyze_image_bytes_creates_crops(tmp_path: Path):
    img_bytes = _make_image_bytes()
    client = DummyClient()
    static_dir = tmp_path / "static"

    result = analyze_image_bytes(img_bytes, client, static_dir=static_dir)

    assert result["image_caption"] == "street with cars"
    assert len(result["objects"]) == 1
    assert result["objects"][0]["caption"] == "a car"

    # Verifica arquivos gravados
    crop_url = result["objects"][0]["crop_url"]
    # traduz URL para caminho
    rel = crop_url.split("/static/")[-1]
    crop_path = static_dir / rel
    assert crop_path.exists()