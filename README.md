# SPECIOUS

**Spectral Perturbation Engine for Contrastive Inference Over Universal Surrogates**

SPECIOUS is a universal, inference-time defence mechanism designed to protect visual artworks from unauthorized AI-driven style mimicry. It generates imperceptible perturbations in the frequency domain, targeting the luminance (Y) channel, to disrupt feature embeddings across various surrogate models without compromising visual quality

---

## Features

* **Model-Agnostic Protection**: Effective across multiple architectures, including ResNet-50 and CLIP ViT-B/32.
* **Label-Free Perturbations**: No need for specific target labels or prompts.
* **Dual-Objective Loss Function**: Balances perceptual fidelity (LPIPS) with feature space distortion.
* **Frequency-Domain Manipulation**: Applies learnable high-pass filters in the Fourier domain.
* **Y-Channel Focused**: Perturbations are confined to the luminance channel, preserving color fidelity.

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/specious.git
   cd specious
   ```

2. **Create and Activate a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare Your Image**:
   Ensure your input image is in a supported format (e.g., JPEG, PNG).

2. **Apply SPECIOUS Perturbation**:

   ```bash
   python apply_specious.py --input path_to_image.jpg --output path_to_output.jpg
   ```

   This script will generate a perturbed image that maintains visual fidelity while disrupting feature embeddings.

3. **Evaluate Perturbation Effectiveness**:
   Use the provided evaluation scripts to assess the impact on surrogate models.

---

## Citation

If you use SPECIOUS in your research or projects, please cite:

```bibtex
@article{your2025specious,
  title={SPECIOUS: Spectral Perturbation Engine for Contrastive Inference Over Universal Surrogates},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025},
  volume={X},
  number={Y},
  pages={Z},
  publisher={Publisher}
}
```



---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information, please contact [pvt.dhruvkumar@gmail.com](mailto:pvt.dhruvkumar@gmail.com).

---

Feel free to customize this `README.md` to better fit your project's specific details and requirements.

[1]: https://arxiv.org/html/2405.19600v1?utm_source=chatgpt.com "Do spectral cues matter in contrast-based graph self-supervised learning?"
