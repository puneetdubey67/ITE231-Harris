import cv2
import numpy as np

cap = cv2.VideoCapture("Raccon.MP4")
if not cap.isOpened():
    print("Error opening video")
    exit()

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Couldn't read first frame")
    exit()

# --------- IMPORTANT FIX ----------
gray_8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # uint8 (good for optical flow)
gray_f32 = np.float32(gray_8)                        # float32 (required for Harris)
# -----------------------------------

# Harris corner detection
block_size = 2
ksize = 3
k = 0.04

harris = cv2.cornerHarris(gray_f32, block_size, ksize, k)
harris = cv2.dilate(harris, None)

threshold = 0.01 * harris.max()

pts = np.argwhere(harris > threshold)
pts = np.flip(pts, axis=1)     # convert [y,x] → [x,y]

p0 = np.float32(pts).reshape(-1, 1, 2)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

print(f"Tracking {len(p0)} points")

# Start processing frames
prev_gray = gray_8.copy()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # NEW FRAME
    gray_8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Lucas-Kanade Optical Flow
    # -------------------------------
    p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_8, p0, None, **lk_params)

    good_new = p1[status == 1]
    good_old = p0[status == 1]

    for new, old in zip(good_new, good_old):
        x_new, y_new = new.ravel()
        x_old, y_old = old.ravel()

        # vector arrow
        cv2.arrowedLine(frame,
                        (int(x_old), int(y_old)),
                        (int(x_new), int(y_new)),
                        (0, 255, 0),
                        2, tipLength=0.4)

        cv2.circle(frame, (int(x_new), int(y_new)), 4, (0, 0, 255), -1)

    cv2.imshow("Harris + Optical Flow Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update
    prev_gray = gray_8.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()