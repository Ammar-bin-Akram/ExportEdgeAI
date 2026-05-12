# post processing functions to be applied on the captured frame
import cv2
import numpy as np


def blur(frame):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(frame, -1, kernel)
    return sharpened


def contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(6,6)) # 1 genertes the same image
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced


def deblur(frame):
    deblurred = cv2.bilateralFilter(frame, 7, 50, 50)
    final = cv2.addWeighted(deblurred, 1.4, cv2.GaussianBlur(deblurred,(3,3),0), -0.4, 0)
    return final


def remove_shadow(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    dilated = cv2.dilate(l, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)

    diff = cv2.subtract(bg, l)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    corrected_l = cv2.subtract(255, norm)

    # Blend original and corrected lightness to avoid overcompensation
    alpha = 0.4  
    blended_l = cv2.addWeighted(l, 1 - alpha, corrected_l, alpha, 0)

    corrected_lab = cv2.merge((blended_l, a, b))
    result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    return result


def smooth_image(frame):
    return cv2.bilateralFilter(frame, d=5, sigmaColor=75, sigmaSpace=75)

