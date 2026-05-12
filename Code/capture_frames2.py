# hsv_motion_hybrid_roi_peak_tflite.py
import cv2
import numpy as np
import os
import time
from collections import deque
from postprocessing import remove_shadow, blur, contrast, deblur, smooth_image
import tensorflow as tf

# =========================
# TFLite Model Setup
# =========================
TFLITE_MODEL_PATH = "models/mobilenet_mango_model.tflite"   # <-- put your .tflite model path here

# defining class indices
class_names = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']

print("-"*10)
print(f"Loading TFLite model from: {TFLITE_MODEL_PATH}")
t0 = time.time()
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
t1 = time.time()
print(f"TFLite model loaded in {t1 - t0:.2f} seconds")
print("-"*10)

def tflite_predict(image_bgr):
    """Runs a single TFLite prediction on a processed image (BGR)."""
    img_resized = cv2.resize(image_bgr, (150, 150))
    img_norm = img_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(img_norm, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start = time.time()
    interpreter.invoke()
    end = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output_data, axis=1)[0]
    confidence = np.max(output_data)

    print(f"Prediction time: {end - start:.3f} seconds")
    print(f"Predicted class: {pred_class} with confidence {confidence:.4f}")
    print(f"Class name: {class_names[pred_class]}")
    return pred_class, confidence

# =========================
# Frame Extraction + Post-processing + Inference
# =========================
# fucntion that captures the video from camera
# captures the main frame from video
# preprocesses it and runs classification model on the frame
# saves result in output folder
def get_frame_from_video(video_path):
    x1, y1, x2, y2 = 162, 31, 498, 465

    MOTION_AREA_THRESHOLD = 5000
    MOTION_END_FRAMES = 10

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
    in_motion = False
    motion_buffer = []
    low_motion_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print("[!] ROI is empty — check coordinates")
            break

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        fgmask = fgbg.apply(roi)
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        _, motion_bin = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_g = cv2.inRange(hsv, lower_green, upper_green)
        mask_b = cv2.inRange(hsv, lower_black, upper_black)
        hsv_mask = cv2.bitwise_or(mask_y, mask_g)
        hsv_mask = cv2.bitwise_or(hsv_mask, mask_b)

        combined = cv2.bitwise_and(motion_bin, hsv_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours)
        print(f"Frame {frame_idx}: motion_area={motion_area:.0f}, in_motion={in_motion}, buffer={len(motion_buffer)}")

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        if not in_motion and motion_area > MOTION_AREA_THRESHOLD:
            in_motion = True
            motion_buffer = []
            low_motion_counter = 0
            print(f"Motion started at frame {frame_idx}")

        if in_motion:
            motion_buffer.append((frame.copy(), motion_area, frame_idx, timestamp_ms))

            if motion_area < MOTION_AREA_THRESHOLD:
                low_motion_counter += 1
            else:
                low_motion_counter = 0

            if low_motion_counter >= MOTION_END_FRAMES:
                in_motion = False
                if motion_buffer:
                    best_frame, best_area, best_idx, best_time = max(motion_buffer, key=lambda x: x[1])
                    roi_crop = best_frame[y1:y2, x1:x2]
                    out_path = f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}ms_area{int(best_area)}.jpg"
                    cv2.imwrite(out_path, roi_crop)
                    saved_count += 1
                    print(f"Saved PEAK frame #{saved_count} (frame {best_idx}, area={best_area:.0f})")

                # Post-processing
                pp_start = time.time()
                image = roi_crop.copy()
                img1 = blur(image)
                img2 = contrast(img1)
                img3 = remove_shadow(img2)
                img4 = deblur(img3)
                final_img = smooth_image(img4)
                pp_end = time.time()
                print(f"Post-processing time: {pp_end - pp_start:.2f} seconds")

                # Fast inference using TFLite
                pred_class, confidence = tflite_predict(final_img)

                # Save intermediate steps (optional)
                cv2.putText(final_img, f"Class: {class_names[pred_class]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(f"{OUTPUT_DIR}/peak_{best_idx:06d}_{best_time}_final.jpg", final_img)

                motion_buffer = []
                in_motion = False
                low_motion_counter = 0

        # Debug visualization
        debug = frame.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Frame (with ROI)", debug)
        cv2.imshow("ROI Combined", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Total saved peak frames:", saved_count)
    return

# =========================
# MAIN
# =========================
VIDEO = "live_recording3.mp4"
OUTPUT_DIR = "relevant_frames3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

get_frame_from_video(VIDEO)
