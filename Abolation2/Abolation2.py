import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define functions for Fourier Transform, Haar Transform, and Anisotropic Gaussian Filter

def fourier_transform(image):
    transformed_image = np.fft.fft2(image)
    magnitude_spectrum = np.abs(transformed_image)
    magnitude_spectrum_scaled = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
    return magnitude_spectrum_scaled


def haar_transform(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_float = np.float32(gray_image) / 255.0
    haar = cv2.dct(image_float)
    transformed_image = np.uint8(haar * 255)
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_GRAY2RGB)
    return transformed_image


def anisotropic_gaussian_filter(image):
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    return filtered_image

# Define function for DnCNN denoising

def dncnn_denoising(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

# Define image denoising function with ablation study

def image_denoising_ablation(image):
    image_for_denoising = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    baseline_denoised_image = image_for_denoising

    transformed_image = fourier_transform(image)
    denoised_with_fourier = dncnn_denoising(transformed_image)

    transformed_image = haar_transform(image)
    denoised_with_haar = dncnn_denoising(transformed_image)

    filtered_image = anisotropic_gaussian_filter(image)
    denoised_with_gaussian = dncnn_denoising(filtered_image)

    denoised_with_dncnn = dncnn_denoising(image_for_denoising)

    return {
        "Baseline": baseline_denoised_image,
        "Fourier Transform": denoised_with_fourier,
        "Haar Transform": denoised_with_haar,
        "Anisotropic Gaussian Filter": denoised_with_gaussian,
        "DnCNN": denoised_with_dncnn
    }


# Load an image locally

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Display the original and denoised images

def display_images(original_image, denoised_images):
    fig, axs = plt.subplots(1, len(denoised_images) + 1, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    for i, (method, denoised_image) in enumerate(denoised_images.items(), start=1):
        axs[i].imshow(denoised_image)
        axs[i].set_title(method)
        axs[i].axis('off')
    plt.show()


# Define function to generate and display graphs

def generate_graphs(image):
    transformed_image = fourier_transform(image)
    haar_image = haar_transform(image)
    gaussian_image = anisotropic_gaussian_filter(image)
    dncnn_image = dncnn_denoising(image)

    titles = ['Fourier Transform', 'Haar Transform', 'Anisotropic Gaussian Filter', 'DnCNN']
    images = [transformed_image, haar_image, gaussian_image, dncnn_image]

    fig, axs = plt.subplots(len(images), 1, figsize=(8, 8 * len(images)))
    for ax, title, img in zip(axs, titles, images):
        ax.plot(img[:, :, 0], label='Channel 1', color='r')
        ax.plot(img[:, :, 1], label='Channel 2', color='g')
        ax.plot(img[:, :, 2], label='Channel 3', color='b')
        ax.set_title(title)
        ax.legend(loc='upper right')  # Manual legend location
    plt.subplots_adjust(top=0.95, bottom=0.05)  # Increase top and bottom margins
    plt.show()



# Example usage

image_path = "C:\\Users\\user\\Desktop\\Final Case\\10% Image\\denoised_image.jpg"  # Provide the path to your image
input_image = load_image(image_path)
denoised_images = image_denoising_ablation(input_image)
display_images(input_image, denoised_images)

# Generate and display graphs
generate_graphs(input_image)
