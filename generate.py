import os
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapOnImage
import numpy as np
import argparse

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def save_images(images, output_folder):
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"augmented_{i}.jpg")
        cv2.imwrite(output_path, image)


def augment_data(images, output_folder, num_augmented_images=270):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Horizontal flip
        iaa.Flipud(0.5),  # Vertical flip
        iaa.Affine(rotate=(-45, 45), scale=(0.5, 1.0)),  # Rotate and scale
        iaa.Multiply((0.8, 1.05))  # Multiply brightness
    ])

    augmented_images = []
    for image in images:
        # imgaug requires images to be in uint8 format
        image = np.clip(image, 0, 255).astype(np.uint8)
        augmented_images.extend(seq(images=[image] * num_augmented_images))

    save_images(augmented_images, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a command-line tool for generating images")
    parser.add_argument("-i", "--input_folder", type=str, help="path/to/your/input/folder", required=True)
    parser.add_argument("-o", "--output_folder", type=str, help="path/to/your/output/folder", required=True)
    parser.add_argument("-n", "--num_augmented_images", type=int, help="number of images to augment", default=10, required=False)
    args = parser.parse_args()

    images = load_images_from_folder(args.input_folder)
    augment_data(images, args.output_folder, args.num_augmented_images)
