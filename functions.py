import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT


# function to display images in cycle
def plot_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# augmentations
def add_augmentations(
    img,
    addBlur=True,
    addShiftScaleRotate=True,
    addRotate=True,
    addHorizontalFlip=True,
    addVerticalFlip=True,
    addTranspose=True,
    addGaussNoise=True,
    addMedianBlur=True,
    addSharpen=True,
    addRandomBrightnessContrast=True,
):

    transformed_imgs = {"initial": img}
    img_tmp = np.array(img, dtype=np.uint8)

    if addBlur:
        # Blur with blur limit 5, 15 or 25
        for blur in [5, 15, 25]:
            transform = A.Blur(blur_limit=(blur, blur + 0.01), always_apply=True)
            transformed_imgs[f"addBlur_{blur}"] = np.array(
                transform(image=img_tmp)["image"], dtype=np.uint8
            )

    # Shift Scale Rotate
    # with angle -28, -12, 12 or 28
    # with shift -0.25 or 0.25
    # with scale 0.1, 0.2 or 0.3
    if addShiftScaleRotate:
        for angle in [-28, -12, 12, 28]:
            for shift in [-0.25, 0.25]:
                for scale in [0.1, 0.2, 0.3]:
                    if angle == 0 and shift == 0 and scale == 0:
                        continue
                    transform = A.ShiftScaleRotate(
                        shift_limit=(shift, shift + 0.01),
                        rotate_limit=(angle, angle + 0.01),
                        scale_limit=(scale, scale + 0.01),
                        always_apply=True,
                    )
                    transformed_imgs[
                        f"addShiftScaleRotate_a{angle}_sh{shift}_sc{scale}"
                    ] = np.array(transform(image=img_tmp)["image"], dtype=np.uint8)

    if addRotate:
        # Rotate 90, 180 or 270
        for angle90 in [90, 180, 270]:
            transform = A.Rotate(limit=(angle90, angle90), p=1)
            transformed_imgs[f"addRotate_{angle90}"] = np.array(
                transform(image=img_tmp)["image"], dtype=np.uint8
            )

    if addHorizontalFlip:
        # Horizontal Flip (True / False)
        transform = A.HorizontalFlip(p=1)
        transformed_imgs[f"addHorizontalFlip"] = np.array(
            transform(image=img_tmp)["image"], dtype=np.uint8
        )

    if addVerticalFlip:
        # Vertical Flip (True / False)
        transform = A.VerticalFlip(p=1)
        transformed_imgs[f"addVerticalFlip"] = np.array(
            transform(image=img_tmp)["image"], dtype=np.uint8
        )

    if addTranspose:
        # Transpose (True / False)
        transform = A.Transpose(p=1)
        transformed_imgs[f"addTranspose"] = np.array(
            transform(image=img_tmp)["image"], dtype=np.uint8
        )

    if addGaussNoise:
        # GaussNoise
        # var_limit from 50 to 4000
        for noise in [50, 500, 1000, 2000, 4000]:
            transform = A.GaussNoise(var_limit=(noise, noise + 0.01), always_apply=True)
            transformed_imgs[f"addGaussNoise_{noise}"] = np.array(
                transform(image=img_tmp)["image"], dtype=np.uint8
            )

    if addMedianBlur:
        # MedianBlur with blur limit 7, 15 or 21 (odd values only)
        for blur in [7, 15, 21]:
            transform = A.MedianBlur(blur_limit=(blur, blur + 2), p=1)
            transformed_imgs[f"addMedianBlur_{blur}"] = np.array(
                transform(image=img_tmp)["image"], dtype=np.uint8
            )

    if addSharpen:
        # Sharpen with alpha 0.2, 0.45 or 0.8
        for alpha1 in [0.2, 0.45, 0.8]:
            transform = A.Sharpen(alpha=(alpha1, alpha1 + 0.01), p=1)
            transformed_imgs[f"addSharpen_{alpha1}"] = np.array(
                transform(image=img_tmp)["image"], dtype=np.uint8
            )

    if addRandomBrightnessContrast:
        # RandomBrightnessContrast
        # with brightness limit -0.4, -0.1, 0.1 or 0.4
        # with contrast_limit -0.4, -0.1, 0.1 or 0.4
        for blimit in [-0.4, -0.1, 0.1, 0.4]:
            for climit in [-0.4, -0.1, 0.1, 0.4]:
                transform = A.RandomBrightnessContrast(
                    brightness_limit=(blimit, blimit + 0.01),
                    contrast_limit=(climit, climit + 0.01),
                    p=1,
                )
                transformed_imgs[
                    f"addRandomBrightnessContrast_b{blimit}_c{climit}"
                ] = np.array(transform(image=img_tmp)["image"], dtype=np.uint8)

    return transformed_imgs


# OpenCV SIFT object
sift = cv2.SIFT_create()
def calc_sift_descriptors(raw_img):
    # normalize images before SIFT
    img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    # sift
    keypoints, descriptors = sift.detectAndCompute(img, None)

    return descriptors, keypoints, img


# skimage SIFT object
descriptor_extractor = SIFT()
def calc_sift_descriptors_skimage(raw_img):
    descriptor_extractor.detect_and_extract(raw_img)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    return descriptors, keypoints
