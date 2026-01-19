# Advance-Face-Recognition

1. Project Structure Should Look Like
Advance-Face-Recognition/
├── people.csv
├── images/
│   ├── rahul.jpg
│   ├── amit.png
│   └── neha.jpeg

2. Install the required libraries
	pip install numpy opencv-python pillow insightface onnxruntime

3. Update people.csv with the name and image_name of the person and keep the image with the same name in the images folder

4. Generate encodings 
	python encode_faces.py

5. Run FR 
	python verify_gui.py

