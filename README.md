# Audio CNN Classifier

A ResNet-style CNN that classifies environmental sounds (50 classes) from mel spectrograms,
with a Next.js dashboard that visualizes the model's internal feature maps.

- Validation accuracy: 84.75% on ESC-50 fold 5
- Backend: PyTorch + Modal (serverless GPU)
- Frontend: Next.js, React, Tailwind


## Features:

- 🧠 Deep Audio CNN for sound classification
- 🧱 ResNet-style architecture with residual blocks
- 🎼 Mel Spectrogram audio-to-image conversion
- 🎛️ Data augmentation with Mixup & Time/Frequency Masking
- ⚡ Serverless GPU inference with Modal
- 📊 Interactive Next.js & React dashboard
- 👁️ Visualization of internal CNN feature maps
- 📈 Real-time audio classification with confidence scores
- 🌊 Waveform and Spectrogram visualization
- 🚀 FastAPI inference endpoint
- ⚙️ Optimized training with AdamW & OneCycleLR scheduler
- 📈 TensorBoard integration for training analysis
- 🛡️ Batch Normalization for stable & fast training
- 🎨 Modern UI with Tailwind CSS & Shadcn UI
- ✅ Pydantic data validation for robust API requests


## Credits
- Andreaswt and ESC-50 dataset by Karol J. Piczak: https://github.com/karolpiczak/ESC-50
  (CC BY-NC 3.0). Non-commercial use only.
