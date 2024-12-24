# Generate images on the test set and download

import cv2
# convert images in black and white
X_test_gray = np.zeros_like(X_test[:, :, :, 0:1])

for i in range(X_test.shape[0]):
    gray_image = cv2.cvtColor(X_test[i], cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image, axis=-1)
    X_test_gray[i] = gray_image
  
# Get model prediction on grey images (change autoencoder if you used another model)
predictions = autoencoder.predict(X_test_gray)
test_loss = autoencoder.evaluate(X_test_gray, X_test, verbose=1)
#create a folder with the colorized images and saved images
output_folder = 'generated_images'
os.makedirs(output_folder, exist_ok=True)

for i, generated_image in enumerate(predictions):
    # Denormalize the image that was previously normalized
    generated_image_denormalized = (generated_image * 255.0).astype(np.uint8)

    # Save image
    filename = os.path.join(output_folder, f'generated_image_{i}.png')
    plt.imsave(filename, generated_image_denormalized.squeeze())

print(f"Generated images saved in the folder: {output_folder}")
#download a zip file of the generated images folder
import shutil

folder_to_zip = 'generated_images'
zip_destination = 'generated_images.zip'
shutil.make_archive(zip_destination[:-4], 'zip', folder_to_zip)
from google.colab import files
files.download(zip_destination)

#Metric
import numpy as np

def get_colourfulness(im):
    """
    Calculate colourfulness in natural images.

    Parameters:
        im: ndarray
            Image in RGB format.

    Returns:
        C: float
            Colourfulness.
    """
    im = im.astype(float)
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    # rg = |R - G|
    rg = np.abs(R - G).flatten()

    # yb = |0.5*(R + G) - B|
    yb = np.abs(0.5 * (R + G) - B).flatten()

    # Standard deviation and mean value of the pixel cloud along directions
    std_RG = np.std(rg)
    mean_RG = np.mean(rg)

    std_YB = np.std(yb)
    mean_YB = np.mean(yb)

    std_RGYB = np.sqrt(std_RG*2 + std_YB*2)
    mean_RGYB = np.sqrt(mean_RG*2 + mean_YB*2)

    C = std_RGYB + (0.3 * mean_RGYB)

    return C

# Upload foders with generated images and original test images as zip file then use this code to unzip them
from zipfile import ZipFile
import os

#generated images
zip_file_path = 'generated_images.zip'
extracted_folder_path = 'content/images'

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

#original test images
zip_file_path = 'test_images.zip'
extracted_folder_path = 'content/test_images'
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

# get colorfullness of all images

folder_path = 'content/images/'
image_files = os.listdir(folder_path)
count=0
colorfullnes=0
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)

    # If the image has 4 channels convert it into rgb
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image_array = np.array(image)
    colorfullnes+=get_colourfulness(image_array)
    count+=1

#print colorfullness and number of images
print(colorfullnes)
print(count)
#define psnr
def psnr(original, compressed):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Parameters:
        original: ndarray
            Original image.
        compressed: ndarray
            Compressed image.

    Returns:
        psnr_value: float
            PSNR value.
    """
    mse = np.mean((original - compressed) ** 2)
    max_pixel = np.max(original)
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value
  #psnr of all images
psnr_tot=0
count=0


folder_path = 'content/images/'
test_images_path = 'content/test_images/test_images/'
png_image_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.png')]

for png_image_file in png_image_files:
    image_number = int(png_image_file.split("_")[-1].split(".")[0])
    png_image_path = os.path.join(folder_path, png_image_file)
    png_image = Image.open(png_image_path)

    # If the image has 4 channels convert it into rgb
    if png_image.mode == 'RGBA':
      png_image = image.convert('RGB')
    png_image_array = np.array(png_image)
    test_image_path = os.path.join(test_images_path, f'image_{image_number + 1}.jpg')
    test_image = Image.open(test_image_path)
    test_image_array = np.array(test_image)

    psnr_tot+=psnr(test_image_array, png_image_array)
    count+=1

#print value
print(psnr_tot)
print(count)
#definte the function to compute ssim
from skimage.metrics import structural_similarity as ssim
def calculate_ssim(original, compressed):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Parameters:
        original: ndarray
            Original image.
        compressed: ndarray
            Compressed image.

    Returns:
        ssim_value: float
            SSIM value.
    """
    ssim_value, _ = ssim(original, compressed, win_size=3, full=True)
    return ssim_value
  #compute ssim on all images

ssim_tot=0
count=0
folder_path = 'content/images/'
test_images_path = 'content/test_images/test_images/'
png_image_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.png')]

for png_image_file in png_image_files:
    image_number = int(png_image_file.split("_")[-1].split(".")[0])
    png_image_path = os.path.join(folder_path, png_image_file)
    png_image = Image.open(png_image_path)

    if png_image.mode == 'RGBA':
      png_image = image.convert('RGB')

    png_image_array = np.array(png_image)
    test_image_path = os.path.join(test_images_path, f'image_{image_number + 1}.jpg')
    test_image = Image.open(test_image_path)
    test_image_array = np.array(test_image)
    ssim_couple=calculate_ssim(test_image_array, png_image_array)
    ssim_tot+=ssim_couple
    count+=1

#print results
print(ssim_tot)
print(count)

# mean squared error

mse_tot=0
count=0
folder_path = 'content/images/'
test_images_path = 'content/test_images/test_images/'
png_image_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.png')]
for png_image_file in png_image_files:
    image_number = int(png_image_file.split("_")[-1].split(".")[0])
    png_image_path = os.path.join(folder_path, png_image_file)
    png_image = Image.open(png_image_path)

    if png_image.mode == 'RGBA':
      png_image = image.convert('RGB')

    png_image_array = np.array(png_image)
    test_image_path = os.path.join(test_images_path, f'image_{image_number + 1}.jpg')
    test_image = Image.open(test_image_path)
    test_image_array = np.array(test_image)
    mse_tot += np.sum((test_image_array - png_image_array) ** 2)
    count+=1

# Compute the mean of the squared errors
mean_mse=mse_tot/count

# Print results
print(mean_mse)
print(count)

#mean absolute errore
mae_tot=0
count=0
folder_path = 'content/images/'
test_images_path = 'content/test_images/test_images/'
png_image_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.png')]

for png_image_file in png_image_files:
    image_number = int(png_image_file.split("_")[-1].split(".")[0])
    png_image_path = os.path.join(folder_path, png_image_file)
    png_image = Image.open(png_image_path)

    if png_image.mode == 'RGBA':
      png_image = image.convert('RGB')

    png_image_array = np.array(png_image)
    test_image_path = os.path.join(test_images_path, f'image_{image_number + 1}.jpg')
    test_image = Image.open(test_image_path)
    test_image_array = np.array(test_image)
    mae_tot += np.sum(np.abs(test_image_array - png_image_array))
    count+=1

# Compute the mean MAE
mean_mae=mae_tot/count
#print results
print(mean_mae)
print(count)

#plot
I upload a folder with selected images for i from 1 to 10 named test_i for the test and autoencoder_i, unet_i, dense_net_i, etc. for the models


import matplotlib.pyplot as plt
from PIL import Image
import os

fig, axs = plt.subplots(8, 10, figsize=(13,12), dpi=200)
for ax in axs.flatten():
    ax.set_yticks([], [])
    ax.set_xticks([], [])
axs[0, 0].set_ylabel("Grayscale\n128x128", rotation=0, fontsize=15, labelpad=60)
axs[1, 0].set_ylabel("Ground truth\n128x128", rotation=0, fontsize=15, labelpad=60)
axs[2, 0].set_ylabel("Autoencoder\n128x128", rotation=0, fontsize=15, labelpad=60)
axs[3, 0].set_ylabel("Deep CNN\n128x128", rotation=0, fontsize=15, labelpad=60)
axs[4, 0].set_ylabel("U-Net\n128x128", rotation=0, fontsize=15, labelpad=60)
axs[5, 0].set_ylabel("ResNet\n128x128", rotation=0, fontsize=15, labelpad=60)
axs[6, 0].set_ylabel("DenseNet\n128x128", rotation=0, fontsize=15, labelpad=60)
axs[7, 0].set_ylabel("Pix2Pix\n128x128", rotation=0, fontsize=15, labelpad=60)
for i in range(10):
    axs[0, i].imshow(Image.open(''.join(["/content/immagini/immagini_finali/test_", str(i+1),".jpg" ])).convert("L"), cmap="gray")
    axs[1, i].imshow(Image.open(''.join(["/content/immagini/immagini_finali/test_", str(i+1),".jpg" ])))
    axs[2, i].imshow(Image.open(''.join(["/content/immagini/immagini_finali/autoencoder_", str(i+1),".png" ])))
    axs[3, i].imshow(Image.open(''.join(["/content/immagini/immagini_finali/deep_nn_", str(i+1),".png" ])))
    axs[4, i].imshow(Image.open(''.join(["/content/immagini/immagini_finali/unet_", str(i+1),".png" ])))
    axs[5, i].imshow(Image.open(''.join(["/content/immagini/immagini_finali/resnet_", str(i+1),".png" ])))
    axs[6, i].imshow(Image.open(''.join(["/content/immagini/immagini_finali/dense_net_", str(i+1),".png" ])))
    axs[7, i].imshow(Image.open(''.join(["/content/immagini/immagini_finali/pix2pix_", str(i+1),".png" ])))
print("finish")
plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()
     
