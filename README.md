# Enhancing EfficientVit with Performer

# Abstract
This project presents an enhanced version of the EfficientViT model aimed at improving scalability, efficiency, and training robustness for image classification tasks. Vision Transformers (ViTs) [1], while powerful, suffer from quadratic complexity in their self-attention mechanism, limiting their de- ployment in real-time and resource-constrained environments. To address this, we replace the standard Multi-Head Self-Attention (MHSA) in EfficientViT [5] with Performer attention [4], a linear-complexity alternative based on kernelized approximations (FAVOR+), enabling faster inference and reduced memory usage. Additionally, we redesign the Feed Forward Network (FFN) using GELU activation [9] and Dropout regularization, enhancing the model’s ability to generalize and resist overfitting. We evaluate the modified architecture on the ImageNet100 [15] dataset and conduct ablation studies to isolate the contributions of each component. These results demonstrate that the enhanced Effi- cientViT model achieves improved accuracy, significantly better computational efficiency, and strong potential for deployment.
# Reference Project
https://github.com/microsoft/Cream/tree/main/EfficientViT/classification
![image](https://github.com/user-attachments/assets/c4d4b153-5686-44c6-88d7-c3756d96c78a)
