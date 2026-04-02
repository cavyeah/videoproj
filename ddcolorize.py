import os
import cv2
import torch
import numpy as np

# -------------------------------
# PATHS
# -------------------------------
input_path  = "input_videos/input2.mp4"
output_path = "output/ddcolor/output2_ddcolor.mp4"
model_path  = "models/ddcolor.pth"

os.makedirs("output/ddcolor", exist_ok=True)

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# LOAD MODEL
# -------------------------------
from basicsr.archs.ddcolor_arch import DDColor

print("Loading DDColor model...")

model = DDColor(
    input_size=256,
    encoder_name='convnext-t',   # lightweight encoder (important)
    decoder_name='MultiScaleColorDecoder',
    num_output_channels=2,
    last_norm='Spectral',
    do_normalize=False,
    num_queries=100,
    num_scales=3,
    dec_layers=9
)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)

model.eval().to(device)

print("✅ Model loaded!")

# -------------------------------
# VIDEO SETUP
# -------------------------------
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise Exception("❌ Cannot open input video")

fps    = cap.get(cv2.CAP_PROP_FPS) or 20
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing: {width}x{height} @ {fps:.1f} FPS")

# -------------------------------
# HELPERS
# -------------------------------
def is_grayscale(frame):
    """Optional skip for already colored frames"""
    b, g, r = cv2.split(frame.astype(np.float32))
    diff = np.mean(np.abs(r - g) + np.abs(r - b) + np.abs(g - b))
    return diff < 12


def process_frame(frame):
    """DDColor inference"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze().cpu().numpy().transpose(1, 2, 0)

    output = (output * 255).clip(0, 255).astype(np.uint8)
    output = cv2.resize(output, (frame.shape[1], frame.shape[0]))

    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


# -------------------------------
# PROCESS LOOP
# -------------------------------
prev_frame = None
alpha = 0.3   # temporal smoothing

frame_count = 0

print("Processing... (ESC to stop)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip already-colored frames (optional)
    if not is_grayscale(frame):
        result = frame
    else:
        result = process_frame(frame)

    # -------------------------------
    # TEMPORAL SMOOTHING
    # -------------------------------
    if prev_frame is not None:
        diff = cv2.absdiff(result, prev_frame)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = mask.astype(np.float32) / 255.0

        result = result * mask[..., None] + prev_frame * (1 - mask[..., None])
        result = result.astype(np.uint8)

    prev_frame = result.copy()

    out.write(result)
    cv2.imshow("DDColor Output", result)

    if frame_count % 30 == 0:
        print(f"→ {frame_count} frames processed")

    if cv2.waitKey(1) == 27:
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Saved to: {output_path}")