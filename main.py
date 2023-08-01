import io
import base64
from typing import Optional, Any
import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
from pydantic import BaseModel


model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", 
load_in_4bit=True, device_map="auto", torch_dtype=torch.float32)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", 
)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Item(BaseModel): 
    imgstring_b64: Optional[str] = None
    prompt: Optional[str] = "Provide a detailed descrition for the image"
    do_sample: Optional[bool] = False
    num_beams: Optional[int] = 5
    max_length: Optional[int] = 256
    min_length: Optional[int] = 1
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.5
    length_penalty: Optional[float] = 1.0
    temperature: Optional[float] = 1.0

def predict(item, run_id, logger):

    item = Item(**item)

    if not item.imgstring_b64:
      logger.info("User did not send image in request")
      return {"status_code": 422, "description": "Please, specify an image"} #returns a 422 status code
    # Do something with parameters from item

    img_bytes = base64.b64decode(item.imgstring_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    inputs = processor(images=image, text=item.prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=getattr(item, "do_sample", False),
        num_beams=getattr(item, "num_beams", 5),
        max_length=getattr(item, "max_length", 256),
        min_length=getattr(item, "min_length", 1),
        top_p=getattr(item, "top_p", 0.9),
        repetition_penalty=getattr(item, "repetition_penalty", 1.5),
        length_penalty=getattr(item, "length_penalty", 1.0),
        temperature=getattr(item, "temperature", 1.0)
    )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    return {"status_code": 200, "description": generated_text}
