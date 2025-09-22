# SPECIOUS [[Paper]](https://snookydru.github.io/SPECIOUS/) [[Working Demo]]()

**Spectral Perturbation Engine for Contrastive Inference Over Universal Surrogates** 

![Web 1920 – 1](https://github.com/user-attachments/assets/50e21349-9056-4007-813a-e6eb5ef23a20)



SPECIOUS is a universal, inference-time defence mechanism designed to protect visual artworks from unauthorized AI-driven style mimicry. It generates imperceptible perturbations in the frequency domain, targeting the luminance (Y) channel, to disrupt feature embeddings across various surrogate models without compromising visual quality


---

## ✨ Features

* **🔒 Model-Agnostic Protection**:
  Robust against diverse architectures such as *ResNet-50* and *CLIP ViT-B/32*.

* **🏷️ Label-Free Perturbations**:
  Generates perturbations without requiring predefined labels or prompts.

* **⚖️ Dual-Objective Loss**:
  Optimizes a balance between perceptual fidelity (*LPIPS*) and feature space distortion.

* **🌐 Frequency-Domain Manipulation**:
  Leverages learnable high-pass filters in the Fourier domain for imperceptible modifications.

* **💡 Y-Channel Focused**:
  Restricts perturbations to the luminance channel, ensuring natural color preservation.


---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/SnookyDru/SPECIOUS.git
   cd specious
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare Your Image**:
   Ensure your input image is in a supported format (e.g., JPEG, PNG, JPG).

2. **Apply SPECIOUS Perturbation**:

   Use run.py script and change the path to the image and trained model's .pt file which you can find in the development folder.

   This script will export the intermediate changes to the image and as well as the final perturbed image in the output folder and also will print the evaluation metrics lik LPIPS, Feature Distortion, etc. in the terminal.

---

## Citation

If you use SPECIOUS in your research or projects, please cite our website: https://snookydru.github.io/SPECIOUS/

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information, please contact [pvt.dhruvkumar@gmail.com](mailto:pvt.dhruvkumar@gmail.com).

