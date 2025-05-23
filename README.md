# Abstract

State-of-the-art Vision Transformer (ViT) models are commonly used for image captioning, where they analyze
visual data and construct captions by embedding connections with specified human inputs. However, this method
restricts these models to text-matching rather than autonomous caption generation, leaving them reliant on
externally provided captions. Furthermore, ViT models have been designed for static image processing and lack
temporal reasoning, making it difficult to comprehend video sequences effectively. Existing video understanding
models seek to analyze videos as a series of static frames, frequently failing to capture motion dynamics and
changing scene settings. Traditional CNN-RNN architectures have been employed for video captioning, but
they face long-term dependencies and complex semantic linkages. Transformers, despite their advantages in
handling extended sequences, are computationally expensive when used directly on videos. In this article, we
present a hybrid approach that combines object detection, frame-wise caption creation, and summarization to
extract relevant insights from video information. Our approach detects essential objects in video frames, provides
contextually rich captions, and then combines them to create a coherent summary that reflects temporal links.
We tested our system on 5 different video samples and found promising gains in automated video understanding
and semantic representation compared to existing image-based captioning models. Our findings indicate that
combining multi-model strategies improves video understanding beyond the constraints of current ViT-based
systems.


## Sections of Video Summary

The project is divided into three sections
### Caption generation
### Dupicate caption removal
### Video summarization from captions

# Approach
The first step is to implement the BLIP model for caption generation. This can be achieved by downloading the dependencies frmo the requirements.txt file

## Program requirements:

transformers
PIL 
torch
cv2
huggingface_hub
sentence_transformers
sklearn
numpy


# Implementing BLIP on videos 
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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


# Filtering Duplicate Captions

captions = [
    "A man is riding a horse.",
    "Someone is on a horse.",
    "The man rides the horse across the field.",
    "A woman is cooking in the kitchen.",
    "a large stadium filled with lots of people",
    "A large stadium packed with a lively crowd watches fireworks light up the sky",
    "Someone is preparing food inside the house."
]
### Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

 embeddings = model.encode(captions)
### Define threshold for similarity (between 0 and 1)
    SIMILARITY_THRESHOLD = 0.7

### Keep unique captions
 unique_captions = []
removed_captions = []
### Track which captions we've already considered
used = set()
for i, emb1 in enumerate(embeddings):
        if i in used:
            continue

unique_captions.append(captions[i])
        
for j in range(i + 1, len(embeddings)):
            if j not in used:
                sim = cosine_similarity([emb1], [embeddings[j]])[0][0]
                if sim >= SIMILARITY_THRESHOLD:
                    removed_captions.append((captions[i], captions[j], sim))
                    used.add(j)



# Finally is the video summary using LlaMa3
## Command to install llama on your local machine
 ollama pull llama3
## Command to start llama 
ollama run llama3
## Prompt to generate summary manually
 prompt ollama run llama3 "Write a fantasy story based on: a man riding a horse, fireworks in a stadium, a child watching the moon."

## Script to automate the generation of video summary
import subprocess

# prompt = """
# Write a story combining:
# - A man is riding a horse.
# - A stadium full of people watching fireworks.
# - A woman cooking dinner.
# """

def get_prompts(capitons):
    # prompt = get_prompts()
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=captions.encode(),
        capture_output=True
    )
    print(result.stdout.decode())
    return result
