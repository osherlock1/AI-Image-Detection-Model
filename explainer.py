#!/usr/bin/env python3
"""
Standalone script to classify an image as AI or Real using CLIP,
generate an attention heatmap, and get a natural language explanation via BLIP-2.
"""
import os
import math
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    Blip2Processor, Blip2ForConditionalGeneration
)
from peft import PeftModel



def get_attention_map(model, processor, image_path, device):
    """
    Returns the original PIL image and a 2D numpy attention map.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.get("pixel_values")
    # Enable attentions
    model.config.output_attentions = True
    outputs = model.vision_model(pixel_values, output_attentions=True)
    attentions = outputs.attentions  # list of (batch, heads, seq, seq)
    # take last layer, batch=0, mean over heads
    last_attn = attentions[-1][0].mean(dim=0)  # (seq, seq)
    # cls token attends to patch tokens
    cls_attn = last_attn[0, 1:]
    num_patches = cls_attn.shape[0]
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        print(f"WARNING: Number of patches ({num_patches}) doesn't match grid ({grid_size}x{grid_size})")
    attn_map = cls_attn.reshape(grid_size, grid_size).detach().cpu().numpy()
    # normalize to [0,1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return image, attn_map

IMAGE_PATH = '/Users/omalleysherlock/Documents/URI_Spring_2025/ELE392_AI_ML/Assignemnets/Final_Project/clip_model/img000013.jpg'
CHECKPOINT_PATH = '/Users/omalleysherlock/Documents/URI_Spring_2025/ELE392_AI_ML/Assignemnets/Final_Project/clip_model/results_unity/final_training_run/final_model'
MODEL_ID = "openai/clip-vit-base-patch32"
BLIP_MODEL_ID = "Salesforce/blip2-flan-t5-xl"
OUTPUT_PATH = "composite.png"

def load_clip(model_id, checkpoint, device):
    processor = CLIPProcessor.from_pretrained(model_id)
    # Load fine-tuned or base model
    if checkpoint and os.path.isdir(checkpoint):
        adapter_config = os.path.join(checkpoint, 'adapter_config.json')
        if os.path.exists(adapter_config):
            print(f"Detected PEFT adapter config at {adapter_config}. Loading base CLIP and adapter.")
            base_model = CLIPModel.from_pretrained(model_id, use_auth_token=True).to(device)
            peft_model = PeftModel.from_pretrained(base_model, checkpoint, use_auth_token=True)
            print("Merging LoRA weights...")
            model = peft_model.merge_and_unload()
            print("LoRA weights merged.")
        else:
            print("Loading full fine-tuned CLIP from", checkpoint)
            model = CLIPModel.from_pretrained(checkpoint, use_auth_token=True)
    else:
        print("Loading base CLIP model", model_id)
        model = CLIPModel.from_pretrained(model_id, use_auth_token=True)
    model = model.to(device)
    model.eval()
    return processor, model

def zero_shot_predict(processor, model, image, device):
    labels = ["AI-generated image", "Real photograph"]
    inputs = processor(
        text=labels, images=image,
        return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
        probs = logits.softmax(dim=1)[0].tolist()
    choice = labels[probs.index(max(probs))]
    print(f"Prediction: {choice} (scores={probs})")
    return choice

def build_composite(image, attn_map, out_path):
    # colorize and resize heatmap
    rgba = plt.get_cmap('viridis')(attn_map)
    arr = (rgba * 255).astype(np.uint8)
    heat = Image.fromarray(arr).convert("RGB")
    heat = heat.resize(image.size, Image.Resampling.LANCZOS)
    # side-by-side
    w, h = image.size
    comp = Image.new("RGB", (2*w, h))
    comp.paste(image, (0,0))
    comp.paste(heat,  (w,0))
    comp.save(out_path)
    print(f"Composite image saved to {out_path}")
    return comp

def explain_with_blip(composite, prediction, blip_model_id, device):
    processor = Blip2Processor.from_pretrained(
        blip_model_id, use_auth_token=True
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        blip_model_id, use_auth_token=True
    ).to(device)
    # Prompt with chain-of-thought reasoning and bullet summary
    prompt = (
        f"The classifier predicted: {prediction}.\n"
        "You are an expert image forensic AI. First, think step by step about why this classification is correct, referencing both the original image and the attention heatmap.\n"
        "After reasoning, provide at least 4 concise bullet points summarizing the key reasons.\n"
        "Begin with 'Reasoning:' on a new line, then 'Summary:' on a new line for the bullets.\n"
    )
    inputs = processor(images=composite, text=prompt, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **inputs,
        max_length=500,
        min_length=200,
        num_beams=4,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
        early_stopping=True
    )
    text = processor.decode(out[0], skip_special_tokens=True)
    print("=== Explanation ===")
    print(text)

if __name__ == "__main__":

    device = "mps" if torch.mps.is_available() else "cpu"
    proc_clip, model_clip = load_clip(
        MODEL_ID, CHECKPOINT_PATH, device
    )
    orig, attn = get_attention_map(
        model_clip, proc_clip, IMAGE_PATH, device
    )
    pred = zero_shot_predict(
        proc_clip, model_clip, orig, device
    )
    comp = build_composite(orig, attn, OUTPUT_PATH)
    print("Generating explanation... this may take a while")
    explain_with_blip(comp, pred, BLIP_MODEL_ID, device)
