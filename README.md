# Audio CNN Sound Classifier

A ResNet-style CNN that classifies environmental sounds (50 classes) from mel spectrograms,
with a Next.js dashboard that visualizes the model's internal feature maps.

- Validation accuracy: 84.75% on ESC-50 fold 5
- Backend: PyTorch + Modal (serverless GPU)
- Frontend: Next.js, React, Tailwind

## Credits
- Trained on the ESC-50 dataset by Karol J. Piczak: https://github.com/karolpiczak/ESC-50
  (CC BY-NC 3.0). Non-commercial use only.