# Self driving car (Small prototype)

Real-time self-driving car simulation system with end-to-end learning for steering prediction and semantic segmentation for scene understanding.

## Features

- **Steering Angle Prediction**: CNN-based model inspired by [NVIDIA's End-to-End Deep Learning paper](https://arxiv.org/pdf/1604.07316)
- **Lane Segmentation**: YOLOv11 model for accurate lane detection
- **Object Detection**: Multi-class object segmentation with bounding boxes
- **Real-time Performance**: 30 FPS inference with parallel processing
- **Visual Feedback**: Live display of segmented scenes and steering wheel rotation

## Architecture

- **Steering Model**: TensorFlow/Keras CNN model
- **Segmentation Models**: YOLOv11 (PyTorch/Ultralytics)
- **Parallel Processing**: Concurrent execution using ThreadPoolExecutor

## Requirements
- tensorflow==2.18.0
- opencv-python
- numpy
- ultralytics

<img width="1483" height="699" alt="Screenshot 2025-10-08 142822" src="https://github.com/user-attachments/assets/cb082298-b477-449a-974e-ea52d2dbf7fd" />

