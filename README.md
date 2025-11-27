# Trained & Pretrained Models Collection

This repository contains standalone training scripts for various deep learning models across different frameworks, specifically designed for medical image classification tasks. Each model can be trained independently and is optimized for Kaggle outsourcing and distributed training.

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision tensorflow keras numpy matplotlib seaborn scikit-learn kagglehub
```

### Training a Model
```bash
# Navigate to the models directory
cd models

# Run any model (example: ResNet-18)
python resnet18_model.py

# Or run a TensorFlow model
python keras_cnn_model.py
```

## 📁 Model Collection

### PyTorch Models

| Model | Framework | Architecture | Use Case |
|-------|-----------|--------------|----------|
| `resnet18_model.py` | PyTorch | ResNet-18 | General image classification |
| `vgg16_model.py` | PyTorch | VGG-16 | Feature-rich classification |
| `densenet121_model.py` | PyTorch | DenseNet-121 | Parameter-efficient classification |
| `efficientnet_b0_model.py` | PyTorch | EfficientNet-B0 | Mobile-optimized classification |
| `mobilenet_v2_model.py` | PyTorch | MobileNetV2 | Lightweight mobile deployment |
| `vit_b_16_model.py` | PyTorch | Vision Transformer | Advanced transformer-based classification |
| `cnn_lstm_model.py` | PyTorch | CNN-LSTM Hybrid | Sequential feature processing |

### TensorFlow/Keras Models

| Model | Framework | Architecture | Use Case |
|-------|-----------|--------------|----------|
| `keras_cnn_model.py` | TensorFlow/Keras | Custom CNN | Custom convolutional networks |
| `keras_resnet_model.py` | TensorFlow/Keras | ResNet-50 | Transfer learning with ResNet |
| `keras_efficientnet_model.py` | TensorFlow/Keras | EfficientNet-B0 | Efficient transfer learning |

## 🏥 Medical Datasets Supported

All models are designed to work with the following medical imaging datasets:

- **Blood Cell Classification**: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL
- **Chest X-ray Pneumonia**: NORMAL vs PNEUMONIA classification
- **Skin Cancer (HAM10000)**: Dermatological lesion classification
- **Brain Tumor Detection**: MRI-based tumor identification

## 🔧 Model Configuration

### Dataset Path
Each script expects the dataset to be in `../dataset/TRAIN/` relative to the script location. You can modify this path:

```python
dataset_path = "../dataset/TRAIN"  # Adjust path as needed
```

### Training Parameters
- **Epochs**: Default 5-10 epochs (configurable)
- **Batch Size**: 32 (configurable)
- **Image Size**: 224x224 (standard for most models)
- **Learning Rate**: 0.001 (Adam optimizer)

### Output Files
Each model saves:
- **PyTorch**: `.pth` files (e.g., `resnet18_model.pth`)
- **TensorFlow**: `.h5` files (e.g., `keras_cnn_model.h5`)
- **Confusion Matrix**: Visual evaluation plots
- **Training Logs**: Console output with accuracy metrics

## 🏗️ Architecture Details

### PyTorch Models
- Built with `torchvision.models` pretrained weights
- Custom classification heads for medical datasets
- Transfer learning with fine-tuning capabilities
- GPU acceleration support

### TensorFlow/Keras Models
- Uses `tensorflow.keras.applications` for transfer learning
- Data augmentation pipelines
- Early stopping and model checkpointing
- Compatible with TensorFlow Serving

## 📊 Performance Metrics

Each model provides:
- Training and validation accuracy/loss curves
- Confusion matrix visualization
- Per-class performance metrics
- Final test accuracy reporting

## 🔄 Kaggle Integration

### For Outsourcing Training:
1. Upload individual model scripts to Kaggle
2. Mount datasets from Kaggle's input section
3. Run training with GPU acceleration
4. Download trained model weights

### Example Kaggle Setup:
```python
# In Kaggle notebook
import os
dataset_path = "/kaggle/input/medical-dataset/TRAIN"
# Modify the dataset_path in the script accordingly
```

## 🤝 Contributing

1. Fork the repository
2. Add new model implementations
3. Test on medical datasets
4. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Repositories

- [Main Transfer Learning Project](https://github.com/Coding-with-Akrash/Transfer-learning) - Complete ensemble application
- [Medical Image Datasets](https://github.com/Coding-with-Akrash/Medical-Imaging-Datasets) - Dataset collection

## 📞 Support

For questions or issues:
- Open an issue in this repository
- Check the main project documentation
- Review individual model comments for implementation details