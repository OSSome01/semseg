# Attention Guided Off-Road Semantic Segmentation

## Problem Statement

Navigating in off-road environments presents a significant challenge for autonomous vehicles due to their unpredictable and noisy nature. Conventional methods often fall short in addressing the complexity of these environments, leading to suboptimal performance and safety concerns. Robust and lightweight methods are needed to accurately identify terrains, obstacles, and objects for safe and efficient navigation. This problem is particularly crucial for applications like search and rescue missions, precision agriculture, and autonomous driving.

## Project Overview

This project focuses on tackling the challenges of off-road environment navigation through Attention Guided Off-Road Semantic Segmentation. The objective is to develop methods capable of accurately identifying and classifying various terrains, obstacles, and objects for improved autonomous vehicle navigation.

## Models Explored

Three different models were compared for semantic segmentation in challenging off-road environments:

1. **U-Net**: Baseline model.
2. **Attention U-Net**: Model enhanced with a visual attention module.
3. **Segformer**: Model chosen for its performance potential.

## Inference Outputs

![Inference Output](https://github.com/OSSome01/semseg/blob/master/assets/Compiled.jpg)

## Key Findings

- The hypothesis that adding a visual attention module enhances the performance of baseline models was validated.
- **Segformer** demonstrated the highest Mean Intersection over Union (mIoU), outperforming both **Attention U-Net** and **U-Net**.
- While **Attention U-Net** performed better than **U-Net**, the performance gap was smaller than anticipated, prompting further investigation.

## Conclusion

In conclusion, this project explored attention-guided semantic segmentation models for off-road environments. The findings supported the effectiveness of incorporating visual attention, with **Segformer** emerging as the top performer. This research contributes to the development of robust and efficient methods for enhancing autonomous vehicle navigation in challenging environments.

