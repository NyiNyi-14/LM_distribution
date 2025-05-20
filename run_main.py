# %% Import lib
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import OrderedDict
from DnCNN import DnCNN
from DCT_IDCT import DCT_IDCT
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
from scipy.spatial.distance import cdist
from collections import defaultdict

# %% Get work dir
import os
os.chdir("")
print(os.getcwd())
print(os.listdir())

# %% Step1.1, Getting frames from video
video_path = ''
output_folder = '' 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_skip = 1 
cap = cv2.VideoCapture(video_path) 
fps = cap.get(cv2.CAP_PROP_FPS)  
start_time = 5
end_time = 12
start_frame = int(start_time * fps)  
end_frame = int(end_time * fps) 

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    if frame_count >= start_frame and frame_count <= end_frame:
        frame_filename = os.path.join(output_folder, f'frame_{saved_count:04d}.png')

        if frame_count % frame_skip == 0:
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames from {start_time} to {end_time} seconds.")

# %% Step1.2, Grayscale + DCT + IDCT
block_size = 8
zero_center = True  # JPEG-like zero-centering
deblocking = DCT_IDCT(block_size = block_size)
DCT_folder = ''

if not os.path.exists(DCT_folder):
    os.makedirs(DCT_folder)
RGB_frames = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])

for frames in RGB_frames:
    RGB_path = os.path.join(output_folder, frames) 
    DCT_path = os.path.join(DCT_folder, frames) 
    img = cv2.imread(RGB_path) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    gray = gray.astype(np.float32)
    if zero_center: # JPEG-like
        gray -= 128  # Zero-center to [-128, 127]
    dct_img, h, w = deblocking.dct2_blockwise(gray) # DCT
    recon_img = deblocking.idct2_blockwise(dct_img, h, w) # IDCT
    if zero_center:
        recon_img += 128 
    recon_img = np.clip(recon_img, 0, 255).astype('uint8')
    cv2.imwrite(DCT_path, recon_img)

print(f"Total {len(RGB_frames)}: Filtered out high freq in DCT and reconstructed with IDCT.")

# %% Step1.3, DnCNN
model = DnCNN(channels=1, num_of_layers=17)
# state_dict = torch.load('models/net_blind.pth', map_location='cpu') # blind noise
# state_dict = torch.load('models/net_n025.pth', map_location='cpu') # n025 noise
state_dict = torch.load('models/net_n050.pth', map_location='cpu') # n050 noise
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") 
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

Denoised_folder = ''
if not os.path.exists(Denoised_folder):
    os.makedirs(Denoised_folder)

DCT_frames = sorted([f for f in os.listdir(DCT_folder) if f.endswith('.png')])

for frames in DCT_frames:
    noise_path = os.path.join(DCT_folder, frames) 
    denoised_path = os.path.join(Denoised_folder, frames) 
    img = cv2.imread(noise_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  

    with torch.no_grad():
        noise = model(img_tensor)
        denoised = img_tensor - noise

    denoised_img = denoised.squeeze().numpy()
    denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(denoised_path, denoised_img)

print(f"Total {len(DCT_frames)}: Denoised")

# %% Step1.4, CLAHE
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
enchanced_folder = ''
if not os.path.exists(enchanced_folder):
    os.makedirs(enchanced_folder)
denoised_file = sorted([f for f in os.listdir(Denoised_folder) if f.endswith('.png')])

for frames in denoised_file:
    input_path = os.path.join(Denoised_folder, frames)
    output_path = os.path.join(enchanced_folder, frames)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) 
    enhanced = clahe.apply(img) 
    cv2.imwrite(output_path, enhanced) 

print(f"Total {len(denoised_file)} Contrast Enhanced.")

# %%  Step2, Edge detection + morphology + contour finding
Morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
enchanced_file = sorted([f for f in os.listdir(enchanced_folder) if f.endswith('.png')])
edges_folder = ''  
morph_folder = ''
contour_folder = ''

os.makedirs(contour_folder, exist_ok=True)
os.makedirs(edges_folder, exist_ok=True)
os.makedirs(morph_folder, exist_ok=True)

count = np.empty(len(enchanced_file), dtype=object)

for i, frames in enumerate(enchanced_file):
    input_path = os.path.join(enchanced_folder, frames)
    output_edge = os.path.join(edges_folder, frames)
    output_morph = os.path.join(morph_folder, frames)
    output_con = os.path.join(contour_folder, frames)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, Morphology_kernel) # dilation followed by erosion
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    droplets = []
    color_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0: 
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                droplets.append({
                    "contour": cnt,
                    "centroid": (cx, cy),
                    "area": area
                })
    count[i] = droplets
    for d in droplets:
        cv2.drawContours(color_img, [d["contour"]], -1, (0, 255, 0), 4)
        cv2.circle(color_img, d["centroid"], 6, (0, 0, 255), -1)

    cv2.imwrite(output_edge, edges)
    cv2.imwrite(output_morph, binary)
    cv2.imwrite(output_con, color_img)

# %% Location and area
location = [
    [droplet['centroid'] for droplet in frame]
    for frame in count
]

area = [[LM['area'] for LM in frame] for frame in count]

# %% Scaling factor
measurement = 0.277  # diameter of the droplet in mm
LM_area = (np.pi * measurement**2) / 4
pixel = area[0][6]
scaling = LM_area / pixel
print(f"Scale factor: {scaling} mm/px")

before = [scaling * a for a in area[0]]
after = [scaling * a for a in area[-1]]

# %% Motion tracking
frame_0000 = location[0]
frame_0070 = location[-1]
centroids_1 = np.array(frame_0000)
centroids_2 = np.array(frame_0070)
D = cdist(centroids_1, centroids_2)

max_dist = 500
matches = {}

for i, row in enumerate(D):
    min_idx = np.argmin(row)
    if row[min_idx] < max_dist:
        matches[i] = min_idx

reverse_matches = defaultdict(list)
for k, v in matches.items():
    reverse_matches[v].append(k)

for new_id, old_ids in reverse_matches.items():
    if len(old_ids) > 1:
        print(f"Droplets {old_ids} merged into LM{new_id}")
    else:
        print(f"Droplet {old_ids[0]} → LM{new_id}")

# %% Visualization 1
fig, ax = plt.subplots(figsize=(8, 6))
for (ind, (x, y)) in enumerate(frame_0000):
    ax.plot(x, y, 'ro', label='before vibrated')
    ax.text(x + 10, y+2, f'LM{ind+1}', color='red', fontsize=16, weight='bold')
for (ind, (x, y)) in enumerate(frame_0070):
    ax.plot(x, y, 'go', label='after vibrated')
    ax.text(x + 10, y+2, f'LM{ind+1}', color='g', fontsize=16, weight='bold')

for i, j in matches.items():
    x1, y1 = frame_0000[i]
    x2, y2 = frame_0070[j]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2),)

ax.invert_yaxis()
ax.set_xlabel(r"$x \ / \ \mathrm{pixel}$", fontsize = 14)
ax.set_ylabel(r"$y \ / \ \mathrm{pixel}$", fontsize = 14)
ax.set_ylim(960, 0)
ax.set_xlim(0, 1280)
# ax.legend(fontsize = 10)
plt.grid(True)
plt.show()

# %% Visualization 2
fig, ax = plt.subplots(figsize=(8, 6))

before_dot, = ax.plot([], [], 'ro', label='before vibrated')
after_dot,  = ax.plot([], [], 'go', label='after vibrated')
arrow_line  = plt.Line2D([0], [0], color='blue', lw=2, label='trajectory')

for (ind, (x, y)) in enumerate(frame_0000):
    ax.plot(x, y, 'ro', markersize = 7)
    ax.text(x + 12, y + 5, rf"$\mathrm{{LM}}_{{{ind+1}}}$", color='red', fontsize=14, weight='bold')

for (ind, (x, y)) in enumerate(frame_0070):
    ax.plot(x, y, 'go', markersize = 7)
    ax.text(x + 20, y - 15, rf"$\mathrm{{LM}}_{{{ind+1}}}$", color='green', fontsize=14, weight='bold')

for i, j in matches.items():
    x1, y1 = frame_0000[i]
    x2, y2 = frame_0070[j]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5, linestyle='--'),)

ax.invert_yaxis()
ax.set_ylim(960, 0)
ax.set_xlim(0, 1280)
ax.grid(True)
ax.set_xlabel(r"$x \ / \ \mathrm{pixel}$", fontsize = 14)
ax.set_ylabel(r"$y \ / \ \mathrm{pixel}$", fontsize = 14)
ax.tick_params(axis='both', labelsize=12)  
ax.legend(handles=[before_dot, after_dot, arrow_line],
          loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=False, ncol=3, fontsize=12)

plt.tight_layout()
# plt.show()
# plt.savefig('', format='pdf', bbox_inches='tight')

# %%