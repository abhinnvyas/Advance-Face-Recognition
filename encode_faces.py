import csv
import os
import cv2
from insightface.app import FaceAnalysis

INPUT_CSV = "people.csv"
IMAGE_DIR = "images"
OUTPUT_CSV = "encodings.csv"
REJECTED_CSV = "rejected.csv"

def is_invalid(val):
    if val is None:
        return True
    val = val.strip().lower()
    return val == "" or val == "undefined" or val == "null"

def main():
    print("[INFO] Loading InsightFace model...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)

    encoded_count = 0
    rejected_count = 0

    with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_enc, \
         open(REJECTED_CSV, "w", newline="", encoding="utf-8") as out_rej:

        reader = csv.DictReader(infile)

        enc_writer = csv.DictWriter(
            out_enc, fieldnames=["name", "image_name", "encoding"]
        )
        rej_writer = csv.DictWriter(
            out_rej, fieldnames=["name", "image_name", "reason"]
        )

        enc_writer.writeheader()
        rej_writer.writeheader()

        for row in reader:
            name = row.get("name")
            image_name = row.get("image_name")

            if is_invalid(name) or is_invalid(image_name):
                rej_writer.writerow({
                    "name": name or "",
                    "image_name": image_name or "",
                    "reason": "Invalid CSV row"
                })
                rejected_count += 1
                continue

            name = name.strip()
            image_name = image_name.strip()
            image_path = os.path.join(IMAGE_DIR, image_name)

            if not os.path.exists(image_path):
                rej_writer.writerow({
                    "name": name,
                    "image_name": image_name,
                    "reason": "Image not found"
                })
                rejected_count += 1
                continue

            img = cv2.imread(image_path)
            if img is None:
                rej_writer.writerow({
                    "name": name,
                    "image_name": image_name,
                    "reason": "Image read error"
                })
                rejected_count += 1
                continue

            faces = app.get(img)
            if len(faces) == 0:
                rej_writer.writerow({
                    "name": name,
                    "image_name": image_name,
                    "reason": "No face detected"
                })
                rejected_count += 1
                continue

            embedding = faces[0].embedding
            embedding_str = ",".join(map(str, embedding.tolist()))

            enc_writer.writerow({
                "name": name,
                "image_name": image_name,
                "encoding": embedding_str
            })

            encoded_count += 1
            print(f"[OK] Encoded: {name}")

    print("\n==============================")
    print(f"✅ Encoded faces : {encoded_count}")
    print(f"❌ Rejected rows : {rejected_count}")
    print("📄 Generated:")
    print("   - encodings.csv")
    print("   - rejected.csv")
    print("==============================")

if __name__ == "__main__":
    main()
