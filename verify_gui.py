import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import csv
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import time
import os

cv2.setLogLevel(0)

ENCODING_CSV = "encodings.csv"
IMAGE_DIR = "images"
MATCH_THRESHOLD = 0.72
MAX_CAMERA_PROBE = 6

known_names = []
known_images = []
known_embeddings = []

latest_frame = None
camera_running = False
camera_thread = None
verifying = False

def load_encodings():
    with open(ENCODING_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            known_names.append(row["name"])
            known_images.append(row["image_name"].strip())
            emb = np.array(list(map(float, row["encoding"].split(","))))
            known_embeddings.append(emb)

    print(f"[INFO] Loaded {len(known_names)} identities")

def list_available_cameras(max_tested=MAX_CAMERA_PROBE):
    cams = []
    for i in range(max_tested):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                cams.append(i)
        c.release()
    return cams

def camera_capture(cam_index):
    global latest_frame, camera_running

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        root.after(0, lambda: messagebox.showerror(
            "Camera Error", f"Cannot open camera {cam_index}"
        ))
        camera_running = False
        return

    while camera_running:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        else:
            time.sleep(0.01)

    cap.release()

def update_ui_frame():
    if latest_frame is not None:
        rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((220, 220))
        imgtk = ImageTk.PhotoImage(img)
        live_label.imgtk = imgtk
        live_label.configure(image=imgtk)

    root.after(30, update_ui_frame)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_frame(frame):
    faces = app.get(frame)
    if not faces:
        return None, "❌ No face detected"

    emb = faces[0].embedding
    scores = [cosine_similarity(emb, e) for e in known_embeddings]
    idx = int(np.argmax(scores))
    score = scores[idx]

    if score >= MATCH_THRESHOLD:
        return idx, f"✅ MATCH\nName: {known_names[idx]}\nScore: {score:.3f}"
    return None, "❌ NO MATCH"

def update_result_ui(idx, text):
    global verifying

    result_label.config(text=text)

    if idx is not None:
        img_path = os.path.join(IMAGE_DIR, known_images[idx])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB").resize((220, 220))
            photo = ImageTk.PhotoImage(img)
            registered_label.config(image=photo)
            registered_label.image = photo
        else:
            registered_label.config(image="")
            registered_label.image = None
    else:
        registered_label.config(image="")
        registered_label.image = None

    verify_button.config(state="normal", text="Verify")
    verifying = False

def verify_async():
    global verifying
    if verifying:
        return

    verifying = True
    root.after(0, lambda: verify_button.config(state="disabled", text="Verifying..."))
    root.after(0, lambda: result_label.config(text=""))

    frame = latest_frame.copy()
    idx, text = verify_frame(frame)

    root.after(0, lambda: update_result_ui(idx, text))

def on_verify_click():
    if latest_frame is None:
        return
    threading.Thread(target=verify_async, daemon=True).start()

def start_camera():
    global camera_running, camera_thread
    stop_camera()
    camera_running = True
    camera_thread = threading.Thread(
        target=camera_capture,
        args=(selected_camera.get(),),
        daemon=True
    )
    camera_thread.start()

def stop_camera():
    global camera_running
    camera_running = False

print("[INFO] Loading InsightFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

dummy = np.zeros((480, 640, 3), dtype=np.uint8)
app.get(dummy)

load_encodings()

root = tk.Tk()
root.title("Face Verification – Live")
root.geometry("560x520")
root.resizable(False, False)

tk.Label(root, text="Live vs Registered", font=("Arial", 16, "bold")).pack(pady=8)

camera_indices = list_available_cameras()
if not camera_indices:
    messagebox.showerror("Error", "No webcams detected")
    root.destroy()
    exit(1)

tk.Label(root, text="Select Camera").pack()
selected_camera = tk.IntVar(value=camera_indices[0])
tk.OptionMenu(root, selected_camera, *camera_indices).pack(pady=5)

frame = tk.Frame(root)
frame.pack(pady=5)

live_label = tk.Label(frame)
live_label.grid(row=0, column=0, padx=10)

registered_label = tk.Label(frame)
registered_label.grid(row=0, column=1, padx=10)

verify_button = tk.Button(root, text="Verify", width=20, command=on_verify_click)
verify_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=5)

selected_camera.trace_add("write", lambda *_: start_camera())

start_camera()
update_ui_frame()

root.protocol("WM_DELETE_WINDOW", lambda: (stop_camera(), root.destroy()))
root.mainloop()
