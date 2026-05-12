# hsv_motion_hybrid_roi_peak.py
import cv2
import numpy as np
import os
from collections import deque
from postprocessing import remove_shadow, blur, contrast, deblur, smooth_image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import time



# function to get the main frame from the video
def get_frame_from_video(video_path, model):
    # x1, y1, x2, y2 = 208, 93, 541, 527   # example — replace with actual ROI
    x1, y1, x2, y2 = 162, 31, 498, 465

    MOTION_AREA_THRESHOLD = 5000
    MOTION_END_FRAMES = 10  # how many consecutive low-motion frames to consider motion ended

    # HSV color ranges
    lower_yellow = np.array([12, 90, 90])
    upper_yellow = np.array([40, 255, 255])
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)

    frame_idx = 0
    saved_count = 0

    # State variables for peak detection
    in_motion = False
    motion_buffer = []  # stores (frame, motion_area, frame_idx, timestamp_ms)
    low_motion_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ROI crop
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print("[!] ROI is empty — check coordinates")
            break

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # motion detection
        fgmask = fgbg.apply(roi)
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        _, motion_bin = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)

        # HSV mask
        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_g = cv2.inRange(hsv, lower_green, upper_green)
        mask_b = cv2.inRange(hsv, lower_black, upper_black)
        hsv_mask = cv2.bitwise_or(mask_y, mask_g)
        hsv_mask = cv2.bitwise_or(hsv_mask, mask_b)

        combined = cv2.bitwise_and(motion_bin, hsv_mask)

        # cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # contour area
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours)
        print(f"Frame {frame_idx}: motion_area={motion_area:.0f}, in_motion={in_motion}, buffer={len(motion_buffer)}")

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Detect start of motion
        if not in_motion and motion_area > MOTION_AREA_THRESHOLD:
            in_motion = True
            motion_buffer = []
            low_motion_counter = 0
            print(f"Motion started at frame {frame_idx}")

        # If in motion, keep buffering frames
        if in_motion:
            motion_buffer.append((frame.copy(), motion_area, frame_idx, timestamp_ms))

            # Check if motion ended
            if motion_area < MOTION_AREA_THRESHOLD:
                low_motion_counter += 1
            else:
                low_motion_counter = 0

            if low_motion_counter >= MOTION_END_FRAMES:
                # Motion event ended → pick peak frame
                in_motion = False
                if motion_buffer:
                    best_frame, best_area, best_idx, best_time = max(motion_buffer, key=lambda x: x[1])
                    roi_crop = best_frame[y1:y2, x1:x2]
                    out_path = f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}ms_area{int(best_area)}.jpg"
                    cv2.imwrite(out_path, roi_crop)
                    saved_count += 1
                    print(f"Saved PEAK frame #{saved_count} (frame {best_idx}, area={best_area:.0f})")
                motion_buffer = []
                in_motion = False
                low_motion_counter = 0

                spt = time.time()
                # applying post processing
                image = roi_crop.copy() # copying the original image

                processed_image = blur(image) # blurring the image
                saved_count += 1

                enhanced_image = contrast(processed_image) # enhancing the image
                saved_count += 1

                shadow_corrected_image = remove_shadow(enhanced_image) # fixing camera shadows
                saved_count += 1

                final_image = deblur(shadow_corrected_image) # deblurring the image
                saved_count += 1

                final_img = smooth_image(final_image) # smoothing the image
                saved_count += 1
                ept = time.time()
                print(f"Post-processing time: {ept-spt:.2f} seconds")

                pst = time.time()
                prediction = model.predict(np.expand_dims(cv2.resize(final_img, (150,150))/255.0, axis=0))
                pet = time.time()
                print(f"Prediction time: {pet-pst:.2f} seconds")
                print(f"Predicted class: {np.argmax(prediction, axis=1)[0]} with confidence {np.max(prediction):.4f}")

                cv2.imwrite(f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}_blur.jpg", processed_image)
                cv2.imwrite(f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}_contrast.jpg", enhanced_image)
                cv2.imwrite(f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}_shadow.jpg", shadow_corrected_image)
                cv2.imwrite(f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}_deblur.jpg", final_image)
                cv2.imwrite(f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}_final.jpg", final_img)

        # Debug visualization
        debug = frame.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Frame (with ROI)", debug)
        cv2.imshow("ROI Combined", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Handle edge case: motion still ongoing at end of video
    if in_motion and motion_buffer:
        best_frame, best_area, best_idx, best_time = max(motion_buffer, key=lambda x: x[1])
        roi_crop = best_frame[y1:y2, x1:x2]
        out_path = f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}ms_area{int(best_area)}.jpg"
        cv2.imwrite(out_path, roi_crop)
        saved_count += 1
        print(f"Saved PEAK frame #{saved_count} (frame {best_idx}, area={best_area:.0f}) — end of video")

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Total saved peak frames:", saved_count)
    return out_path



""" main loop """
# VIDEO = "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"

VIDEO = "live_recording1.mp4"
OUTPUT_DIR = "relevant_frames1"
os.makedirs(OUTPUT_DIR, exist_ok=True) # making output directory


num_classes = 5
# loading the model
base_model = tf.keras.applications.MobileNetV2(input_shape=(150,150,3),
                         include_top=False,  # exclude the original FC layers
                         weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)   # helps reduce overfitting
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(x)

model = Model(inputs=base_model.input, outputs=predictions)

MODEL_PATH = "models\mobilenet_mango_model.h5"
print("-"*10)
print("Loading model weights from:", MODEL_PATH)
st = time.time()
model.load_weights(MODEL_PATH)
et = time.time()
print(f"Model fully loaded in {et-st:.2f} seconds")
print("-"*10)


# getting the frame saved
frame_path = get_frame_from_video(VIDEO, model)


