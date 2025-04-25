from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
from ultralytics import YOLO
import cv2
from huggingface_hub import login
import threading
import time
import os


lock = threading.Lock()

API_KEY = 'get_api_key_for_gated_models'
login(token=API_KEY)
cap = cv2.VideoCapture('path/to/video')

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

captions = []

def get_captions():
    global captions
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (480, 480))
        # cv2.imshow('frame', frame) you can choose to show video frames
        if counter >= 10:
            continue  
        inputs = processor(frame, return_tensors="pt")
        caption_ids = blip_model.generate(**inputs)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
        captions.append(caption)
        print(caption)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        # time.sleep(2) to control I/O
        counter+=1
        print(counter)
    return captions

from check_meaning import get_unique_captions

all_captions = get_captions()
#get unique captions
unique_captions = get_unique_captions(all_captions)


from local_llm import get_prompts
#final video summary from llama3
get_prompts(unique_captions)

print(get_prompts)
