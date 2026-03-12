import cv2
import numpy as np
import os
import glob

# Canny thresholds - tried a few values, 50/150 gave the cleanest edges
CANNY_LOW = 50
CANNY_HIGH = 150

# Hough transform params - tuned these to reduce false detections
HOUGH_THRESHOLD = 30
MIN_LINE_LEN = 50
MAX_LINE_GAP = 100

# only look at the bottom 40% of the image (where the road actually is)
ROI_TOP = 0.60

# draw lane center point (extra credit)
SHOW_CENTER = True


def get_roi(edges, shape):
    h, w = shape[:2]
    top_y = int(h * ROI_TOP)

    # trapezoid shaped mask to focus on the road ahead
    pts = np.array([[
        (0, h),
        (w, h),
        (int(w * 0.55), top_y),
        (int(w * 0.45), top_y)
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(edges, mask)


def average_lines(segments, shape):
    # take all line segments on one side and average them into a single line
    if not segments:
        return None

    h = shape[0]
    top_y = int(h * ROI_TOP)

    all_x, all_y = [], []
    for x1, y1, x2, y2 in segments:
        all_x += [x1, x2]
        all_y += [y1, y2]

    # fit x = f(y) instead of y = f(x) to avoid issues with steep lines
    try:
        coeffs = np.polyfit(all_y, all_x, 1)
    except:
        return None

    f = np.poly1d(coeffs)
    return (int(f(h)), h, int(f(top_y)), top_y)


def split_left_right(lines, img_width):
    left, right = [], []
    mid = img_width / 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)

        # ignore nearly flat lines, they're usually noise
        if abs(slope) < 0.3:
            continue

        if slope < 0 and x1 < mid and x2 < mid:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > mid and x2 > mid:
            right.append((x1, y1, x2, y2))

    return left, right


def draw_lanes(img, left, right):
    out = img.copy()

    for lane in [left, right]:
        if lane is None:
            continue
        x1, y1, x2, y2 = lane
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 8)

    # extra credit: mark the center between the two lanes
    if SHOW_CENTER and left is not None and right is not None:
        cx = (left[0] + right[0]) // 2
        cy = img.shape[0] - 10
        cv2.circle(out, (cx, cy), 15, (0, 0, 255), -1)
        cv2.putText(out, "Center", (cx - 30, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return out


def process_image(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Could not read {img_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    roi = get_roi(edges, img.shape)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180,
                            threshold=HOUGH_THRESHOLD,
                            minLineLength=MIN_LINE_LEN,
                            maxLineGap=MAX_LINE_GAP)

    left_line = right_line = None
    if lines is not None:
        left_segs, right_segs = split_left_right(lines, img.shape[1])
        left_line = average_lines(left_segs, img.shape)
        right_line = average_lines(right_segs, img.shape)

    result = draw_lanes(img, left_line, right_line)
    cv2.imwrite(out_path, result)

    n = sum(x is not None for x in [left_line, right_line])
    print(f"  {out_path} -> {n} lane(s) detected")
    return True


def main():
    os.makedirs("output", exist_ok=True)

    image_paths = sorted(glob.glob("images/*.jpg") +
                         glob.glob("images/*.jpeg") +
                         glob.glob("images/*.png"))

    if not image_paths:
        print("No images found in images/ folder.")
        return

    image_paths = image_paths[:10]
    print(f"Running lane detection on {len(image_paths)} images...\n")

    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = f"output/{name}_lanes.jpg"
        process_image(img_path, out_path)

    print("\nDone! Results saved in output/")


if __name__ == "__main__":
    main()
