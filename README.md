# Predictive Beauty Analytics: Foundation Shade Recommendation
An end-to-end data science, computer vision, and machine learning project that estimates skin tone from facial images and recommends foundation shades using perceptual color similarity.
The project combines MediaPipe facial landmark detection, CIELAB color analysis, CIEDE2000 color distance, K-Means clustering, and model evaluation to build and assess a foundation shade recommendation pipeline.
## Project Overview
Choosing a foundation shade online is difficult because product shade names are inconsistent across brands and photographs can be affected by lighting, cameras, and skin-tone variation.
This project explores a data-driven approach that:
1. Detects a face from an input image.
2. Extracts representative skin-color regions.
3. Converts skin and foundation colors into the CIELAB color space.
4. Measures perceptual similarity using CIEDE2000.
5. Ranks the closest available foundation shades.
6. Evaluates pipeline robustness across diverse facial images.
7. Uses clustering and review-based machine learning experiments to explore additional recommendation signals.

## Pipeline
```text
Input Selfie
    ↓
Face Detection & Facial Landmarks
    ↓
Skin Region Extraction
    ↓
RGB → CIELAB Conversion
    ↓
Skin Tone Estimation
    ↓
CIEDE2000 Color Distance
    ↓
Foundation Shade Ranking
    ↓
Top-N Recommendations

```
