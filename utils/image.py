import io
import tempfile
from PIL import Image

def compress_image(image_bytes: bytes, quality: int = 75) -> bytes:
    image = Image.open(io.BytesIO(image_bytes))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()