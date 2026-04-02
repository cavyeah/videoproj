import cv2
import numpy as np
import torch
import struct
from colorizers import siggraph17, preprocess_img, postprocess_tens

bits = struct.calcsize('P') * 8
print(f"Python: {bits}-bit")
HW = (256, 256)
torch.set_num_threads(4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

input_path  = "input_videos/input2.mp4"
output_path = "output/output2.mp4"
weights     = r"C:\Users\HP\.cache\torch\hub\checkpoints\siggraph17-df00044c.pth"

print("Loading model...")
colorizer = siggraph17(pretrained=False).eval()
colorizer.load_state_dict(torch.load(weights, map_location=device))
colorizer = colorizer.to(device)
print("✅ Model loaded!")

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise Exception("❌ Cannot open input video")

fps    = cap.get(cv2.CAP_PROP_FPS) or 20
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height} @ {fps:.1f}fps")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

sharpen_kernel = np.array([[0, -1,  0],
                            [-1,  5, -1],
                            [0, -1,  0]])

prev_frame  = None
frame_count = 0
alpha       = 0.15

def is_grayscale_region(region):
    b, g, r = cv2.split(region)
    diff_rg = cv2.absdiff(r, g).mean()
    diff_rb = cv2.absdiff(r, b).mean()
    diff_gb = cv2.absdiff(g, b).mean()
    return diff_rg < 20 and diff_rb < 20 and diff_gb < 20

def colorize_region(region):
    img_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tens_l_orig, tens_l_rs = preprocess_img(img_rgb, HW=HW)
    tens_l_rs = tens_l_rs.to(device)
    with torch.no_grad():
        out_tens = colorizer(tens_l_rs)
    out_tens = out_tens.cpu()
    result = postprocess_tens(tens_l_orig, out_tens, mode='bilinear')
    result_bgr = cv2.cvtColor(
        (np.clip(result, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )
    result_bgr = cv2.resize(result_bgr, (region.shape[1], region.shape[0]))
    result_bgr = cv2.filter2D(result_bgr, -1, sharpen_kernel)
    return result_bgr

def boost_saturation(frame, scale=1.4):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] = np.clip((lab[:, :, 1] - 128) * scale + 128, 0, 255)
    lab[:, :, 2] = np.clip((lab[:, :, 2] - 128) * scale + 128, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

print("Processing... (press ESC to stop early)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    mid   = width // 2
    left  = frame[:, :mid]
    right = frame[:, mid:]

    left_bw  = is_grayscale_region(left)
    right_bw = is_grayscale_region(right)

    if left_bw and right_bw:
        result_bgr = colorize_region(frame)
        result_bgr = boost_saturation(result_bgr, scale=1.4)

    elif not left_bw and not right_bw:
        result_bgr = frame

    else:
        if left_bw:
            left_out  = boost_saturation(colorize_region(left), scale=1.4)
            right_out = right
        else:
            left_out  = left
            right_out = boost_saturation(colorize_region(right), scale=1.4)
        result_bgr = np.concatenate([left_out, right_out], axis=1)

    if prev_frame is not None:
        result_bgr = cv2.addWeighted(result_bgr, 1 - alpha, prev_frame, alpha, 0)
    prev_frame = result_bgr.copy()

    out.write(result_bgr)
    cv2.imshow("Output", result_bgr)

    if frame_count % 30 == 0:
        print(f"  → {frame_count} frames processed...")

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Done! {frame_count} frames → {output_path}")