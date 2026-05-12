import cv2
import datetime

# RTSP URL of your Dahua camera (use subtype=1 for lighter stream)
url = "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"

# Open video capture
cap = cv2.VideoCapture(url)

# Check if opened
if not cap.isOpened():
    print("Failed to open camera stream")
    exit()

# Get stream properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 20  # fallback to 20 if not detected

print(f"Resolution: {width}x{height}, FPS: {fps}")

# Define codec and VideoWriter for recording
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # use 'MP4V' for .mp4
out = cv2.VideoWriter("live_recording_all.mp4", fourcc, fps, (width, height))

snapshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Show live feed
    cv2.imshow("Dahua Live Feed", frame)

    # Write video frame
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' → save snapshot
    if key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")
        snapshot_count += 1

    # Press 'q' → quit
    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
