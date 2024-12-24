# Image Colorization with Deep Learning Models

This project aims to automate the process of colorizing grayscale images using deep learning techniques, focusing specifically on the Pix2Pix architecture. The task is inherently challenging due to the ill-posed nature of image colorization, where multiple plausible colorings exist for a given grayscale image. In this study, we compare several neural network architectures, including traditional models and more advanced ones like Pix2Pix.

## Overview

The goal of the image colorization task is to predict plausible color channels for a given grayscale image, turning it into a realistic colored version. While early approaches relied on user guidance (e.g., color scribbles or reference images), deep learning models, such as those we experiment with in this project, enable fully automated colorization.

### Models Used

This project evaluates the effectiveness of different neural network architectures in the task of image colorization, including:

- **Convolutional Auto-Encoder**
- **Deep CNN**
- **U-Net**
- **ResNet**
- **DenseNet**
- **Pix2Pix (cGAN)**

### Dataset

We use a comprehensive dataset of over 8000 images of flowers from 102 different species. These images are grayscale and have been used to train and test the various models described above.

## Key Contributions

- **Traditional Neural Networks**: We started with simpler architectures, including deep neural networks, autoencoders, and basic convolutional networks.
- **Advanced Models**: We progressively moved to more sophisticated models like U-Net, ResNet, DenseNet, and ultimately Pix2Pix, to better handle the complexity of the colorization task.
- **Evaluation Metrics**: We evaluate the models using several metrics, including:
  - **MSE (Mean Squared Error)**
  - **MAE (Mean Absolute Error)**
  - **PSNR (Peak Signal-to-Noise Ratio)**
  - **SSIM (Structural Similarity Index)**
  - **Colorfulness (a metric proposed by Hasler et al.)**

### Key Observations

- **Pix2Pix and U-Net**: These models performed best in terms of visual quality, as they were able to preserve fine details and generate realistic colors.
- **ResNet, DenseNet**: These models showed solid performance, but their results were less refined compared to Pix2Pix and U-Net in terms of texture and details.
- **Metrics**: No single model outperformed others across all metrics, due to the inherent ambiguity in colorization tasks.

## Architecture Details

### Convolutional Auto-Encoder

The autoencoder model was the first architecture tested. It consists of an encoder-decoder structure with convolutional layers designed to extract and reconstruct image features. The encoder compresses the grayscale input into a compact representation, while the decoder reconstructs the colorized output.

### Deep CNN

We designed a custom deep Convolutional Neural Network for this task, specifically tailored for grayscale images of 128x128 pixels. The architecture includes multiple convolutional layers with skip connections after every three layers to preserve spatial features.

### U-Net

Our U-Net implementation draws inspiration from the original architecture, utilizing both down-sampling and up-sampling paths. Residual layers are incorporated within the down-sampling section to improve feature extraction, while up-sampling layers help retain information during the reconstruction phase.

### ResNet

ResNet, known for its deep residual learning structure, was employed to handle the complex mapping of grayscale to colorized images. The architecture includes multiple residual blocks, helping alleviate the vanishing gradient problem in deeper networks.

### DenseNet

This architecture utilizes dense blocks to improve gradient flow and feature reuse. It helps in learning more intricate features, which is crucial for the colorization task.

### Pix2Pix

Pix2Pix is a conditional Generative Adversarial Network (cGAN) used for image-to-image translation tasks, including colorization. It consists of a generator (which learns the mapping from grayscale to color) and a discriminator (which distinguishes between real and generated images). The generator follows an encoder-decoder structure with skip connections similar to U-Net.

## Loss Functions

- **MSE Loss**: Used in traditional networks to measure the pixel-wise difference between predicted and ground truth color images.
- **Adversarial Loss**: Used in Pix2Pix, this combines both **MSE** and **Binary Cross-Entropy (BCE)** losses to encourage the generator to create realistic images.
- **Reconstruction Loss**: Measures the pixel-wise difference between the generated and real RGB images, guiding the generator to accurately reconstruct the colorized images.

## Evaluation

We evaluated the models with several metrics to assess both the quality and realism of the colorized images:

- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **Colorfulness**: A metric measuring the vibrancy of colors in the image.

### Performance Summary (Table 1)

| Model       | Colorfulness | PSNR   | SSIM  | MSE        | MAE        |
|-------------|--------------|--------|-------|------------|------------|
| Autoenc     | 15916.6      | 45697.3| 347.5 | 5081997    | 5885808    |
| Deep NN     | 22045.0      | 45700.4| 428.8 | 5080131    | 5976311    |
| Dense Net   | 20934.3      | 45683.7| 377.0 | 5091630    | 5902259    |
| ResNet      | 21575.1      | 45700.1| 401.7 | 5080246    | 5945178    |
| U-Net       | 20723.7      | 45657.8| 434.7 | 5061775    | 5896118    |
| Pix2Pix     | 16742.3      | 45695.1| 429.0 | 5083569    | 6003876    |

## Conclusion

This project demonstrates the comparative performance of several deep learning models for the challenging task of image colorization. While no single model is superior in all metrics, Pix2Pix and U-Net show promising results in terms of visual fidelity and realism.

By employing state-of-the-art neural architectures, we are able to significantly improve the quality of colorized images compared to traditional methods.

---

For further details and implementation, check out the provided code and scripts in this repository.

## Authors:
-Bazzan Sofia
-Pasin Diletta
