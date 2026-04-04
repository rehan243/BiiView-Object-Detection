---

# BiiView — Object Detection for Interactive Shopping

Real-time object detection in video using OpenCV and Meta AI's Segment Anything Model (SAM), enabling direct product identification and shopping from video feeds.

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)

---

## Overview

BiiView transforms passive video viewing into an interactive shopping experience. Using Meta AI's Segment Anything Model (SAM) combined with custom OpenCV pipelines, the system detects and segments objects in real-time video feeds, allowing users to identify and shop for products directly from what they see on screen.

Developed at **Verticiti**, handling a massive dataset of 11 million images and 1.1 billion segmentation masks, achieving 90% segmentation accuracy.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Video Input                         │
│  Live stream / recorded video / camera feed           │
└─────────────────────────┬────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│              Frame Extraction Pipeline                  │
│  - Keyframe detection                                 │
│  - Frame rate optimization                            │
│  - Resolution normalization                           │
└─────────────────────────┬────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│         Segment Anything Model (SAM)                   │
│  - Automatic mask generation                          │
│  - Point/box prompt segmentation                      │
│  - Zero-shot object segmentation                      │
│  - 11M+ images trained, 1.1B+ masks                  │
└─────────────────────────┬────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │                               │
┌─────────▼──────────┐   ┌───────────────▼────────────┐
│  Object Detection  │   │  Instance Segmentation      │
│  - YOLO/Faster     │   │  - Pixel-level masks        │
│    R-CNN backup    │   │  - Object boundaries        │
│  - Class labels    │   │  - Multi-object tracking    │
└─────────┬──────────┘   └───────────────┬────────────┘
          │                               │
┌─────────▼───────────────────────────────▼───────────┐
│           Product Matching Engine                     │
│  - Visual similarity search                          │
│  - Product catalog matching                          │
│  - Embedding-based retrieval                         │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│            Interactive Shopping Layer                  │
│  - Clickable product overlays                        │
│  - Real-time product cards                           │
│  - Purchase links                                    │
└─────────────────────────────────────────────────────┘
```

## Key Features

- **Segment Anything Model**: Meta AI's SAM for zero-shot, high-precision object segmentation across any visual domain
- **90% Accuracy**: Object detection and segmentation accuracy across diverse product categories
- **11M+ Images**: Trained and evaluated on massive dataset with 1.1 billion segmentation masks
- **Real-Time Processing**: Optimized inference pipeline for live video feeds
- **Interactive Shopping**: Users click detected objects to view product details and purchase links
- **Multi-Object Tracking**: Track multiple objects across video frames with consistent IDs
- **Product Matching**: Visual similarity search to match detected objects to product catalogs

## Tech Stack

| Category | Technologies |
|---|---|
| **Segmentation** | Meta AI SAM, SAM-HQ |
| **Detection** | YOLOv8, Faster R-CNN |
| **Computer Vision** | OpenCV, torchvision |
| **Deep Learning** | PyTorch, CUDA |
| **Embeddings** | CLIP, ResNet features |
| **API** | FastAPI, WebSockets |
| **Frontend** | JavaScript, Canvas API |
| **Infrastructure** | Docker, GPU inference |

## Results

| Metric | Value |
|---|---|
| Segmentation accuracy | 90% |
| Images processed | 11M+ |
| Segmentation masks | 1.1B+ |
| Inference latency (per frame) | < 100ms |
| Objects tracked per frame | 50+ |
| Product match accuracy | 85% |

---

> **Source Code**: The production source code for this project is maintained in a private repository due to proprietary and client confidentiality requirements. This repository documents the architecture, design decisions, and technical approach. For code-level discussions or collaboration inquiries, feel free to reach out.


## Author

**Rehan Malik** — Senior AI/ML Engineer @ Reallytics.ai

- [LinkedIn](https://linkedin.com/in/rehan-malik-62b3301ab)
- [GitHub](https://github.com/rehan243)

---