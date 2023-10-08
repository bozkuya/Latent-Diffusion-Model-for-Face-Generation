# Latent Diffusion Model for Face Generation

## Description
This project implements a latent diffusion model for generating highly realistic facial images. The model first projects input images to a latent space using an autoencoder and then trains a diffusion model on this latent space. We trained our model using the FFHQ dataset and fine-tuned it using a specialized dataset of LeBron James.

## Installation

### Prerequisites
- Python 3.7 or later
- Git LFS

### Steps
1. Clone this repository:
```bash
git clone <https://github.com/bozkuya/Latent-Diffusion-Model-for-Face-Generation>
```
2. Change to the directory of the cloned repository:
```bash
   cd <repository-directory>
```

### Training
1. Download the FFHQ dataset using the provided [Kaggle link](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only) and extract it to the `data/` folder.
2. Use the `AutoencoderKL` class from the Diffusers library to project the images to a latent space.
3. Train the diffusion model using the `UNet2DModel` class from the Diffusers library.

### Generation
Use the trained model to generate new face samples and visualize them. The code provides a facility to do this using the DDIM sampler for faster and deterministic sampling.

### Fine-tuning
The model can be fine-tuned on a single subject using the Dreambooth method, as demonstrated with a dataset of LeBron James in the provided [Colab notebook](https://colab.research.google.com/drive/1Gk64JKvC8gNR6uoYs3CY5HCYMi6m0QMi).

### Sample Code Execution
Follow the steps in the provided Colab notebook, [Untitled0.ipynb](https://colab.research.google.com/drive/1Gk64JKvC8gNR6uoYs3CY5HCYMi6m0QMi), to understand the complete process, from setting up the environment to training and fine-tuning the model.

## Acknowledgments
- FFHQ dataset is available on [Kaggle](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only).
- The autoencoder model is available on [Hugging Face](https://huggingface.co/stabilityai/sdxl-vae).
- Training code example and other resources can be found on [Hugging Face's Diffusers repository](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py).

