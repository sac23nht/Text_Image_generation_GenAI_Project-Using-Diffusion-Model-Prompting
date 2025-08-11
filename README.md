# Diffusion Meme Generator with LoRA Fine-Tuning on Maya Dog Dataset

This repository contains a Streamlit app that generates images using the **Stable Diffusion v1.4** text-to-image model, enhanced with LoRA fine-tuning on a custom dataset of Currome Maya dog images. The app allows you to generate AI-powered images and overlay meme-style text on them.

---
<img width="1355" height="663" alt="image" src="https://github.com/user-attachments/assets/65535c87-847d-4bd9-b930-4b56135dc96f" />

## Table of Contents

- Overview  
- Model Details  
- Building Your Own Pipeline  
- Using the Pretrained Stable Diffusion v1.4 Pipeline  
- Fine-Tuning Stable Diffusion with LoRA on Currome Maya Dog Images  
- Running the Streamlit App  
- Requirements  
- License

---

## Overview

Stable Diffusion v1.4 is a state-of-the-art latent diffusion model that generates high-quality images from text prompts. This project demonstrates:

- How to build and customize diffusion pipelines  
- How to use a pretrained Stable Diffusion v1.4 pipeline for image generation  
- How to fine-tune the model efficiently using **LoRA (Low-Rank Adaptation)** on a custom dataset of Maya dog images  
- How to generate meme-style images with customizable text overlays using Streamlit

---

## Model Details

### Stable Diffusion v1.4

- Developed by CompVis, Stable Diffusion v1.4 is a latent text-to-image diffusion model  
- It uses a U-Net backbone for denoising and a CLIP-based text encoder for prompt understanding  
- Generates 512x512 pixel images conditioned on text inputs

### LoRA (Low-Rank Adaptation)

- LoRA is a parameter-efficient fine-tuning technique that adapts large models by injecting low-rank update matrices into model weights  
- This method requires significantly fewer parameters and less compute than full fine-tuning  
- In this project, LoRA weights were trained on the Currome Maya dog dataset, allowing the model to learn new dog-specific features and styles without retraining the entire network  

---

## Building Your Own Pipeline

To build a diffusion pipeline from scratch, you need to:

1. Define core components such as the U-Net denoiser, variational autoencoder (VAE), text encoder (e.g., CLIP), and scheduler  
2. Assemble these components into a Stable Diffusion pipeline using libraries like Hugging Face Diffusers  
3. Optionally train or fine-tune the pipeline on your own dataset

This approach gives you full control over model architecture and training.

---

## Using the Pretrained Stable Diffusion v1.4 Pipeline

For most use cases, you can load the pretrained Stable Diffusion v1.4 model from Hugging Face:

- The pipeline includes all model components pretrained on large-scale datasets  
- Supports efficient generation on CPU or GPU  
- Can be extended with LoRA weights for domain adaptation

---

## Fine-Tuning Stable Diffusion with LoRA on Currome Maya Dog Images

### Dataset: Currome Maya Dog Images

- A collection of images featuring Maya dogs with diverse poses and backgrounds  
- Used to teach the model fine-grained concepts about this dog breed

### Fine-Tuning Steps

1. Prepare paired images and text prompts describing the Maya dog features  
2. Use LoRA training scripts (not included in this repo) to fine-tune the Stable Diffusion weights with your dataset  
3. Save the resulting LoRA weights in `pytorch_lora_weights.safetensors` format  
4. Load these LoRA weights during inference to enhance the pretrained pipeline with Maya dog knowledge  

This method is lightweight and modular, enabling fast experimentation and incremental learning.

---

## Running the Streamlit App


The app provides an interactive interface to:

- Enter a text prompt describing the desired image  
- Specify the number of images to generate (up to 10)  
- Add custom text overlays (meme captions) on generated images  

### Usage Instructions

1. Install dependencies listed below  
2. Run the app with `streamlit run app.py`  
3. Input prompt and overlay text in the sidebar  
4. Click “Generate Images” to produce and display images

### Features

- GPU acceleration support if CUDA is available  
- Automatic loading of pretrained pipeline and LoRA weights  
- Text overlay rendering with adjustable font and outline for meme creation

---

## Requirements

- Python 3.8+  
- PyTorch  
- diffusers (Hugging Face)  
- streamlit  
- pillow  

Install dependencies with:



Install dependencies via pip before running the app.


Feel free to contribute or customize the pipeline for your own use cases!
Contact Details:
saijaya.c@gmail.com

---

**Note:** This repository assumes familiarity with Stable Diffusion, LoRA fine-tuning, and Python development. For detailed LoRA training scripts and dataset preparation, please refer to the official diffusers documentation and LoRA repositories.

---
