# Real-Time-Finger-Counter-with-OpenCV


# ✋ Real-Time Finger Counter with OpenCV

This project is a Python application that uses a webcam to detect and count the number of fingers held up in real time. It uses background subtraction, image segmentation, and contour analysis to achieve finger counting from a live video feed.

## 🧠 Features

- Real-time hand and finger detection using OpenCV
- Background averaging for accurate segmentation
- Convex hull analysis and circular ROI to detect extended fingers
- Clean and well-commented code for learning purposes
- Easily extendable for gesture control applications

---

## 📦 Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- scikit-learn (for `pairwise.euclidean_distances`)

Install the dependencies:

```bash
pip install opencv-python numpy scikit-learn
```


## 🚀 How It Works

1. **Background Model Initialization:**  
   For the first 60 frames, the application builds a running average of the background to use for hand segmentation.

2. **Hand Segmentation:**  
   The hand is segmented from the background using frame differencing and thresholding.

3. **Contour & Convex Hull Analysis:**  
   The largest contour is assumed to be the hand. Convex hull and extreme points are used to estimate hand center.

4. **Finger Counting:**  
   A circular region around the palm center helps count the number of finger-like contours, ignoring wrist and noise.

---

## 🖥️ Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/finger-counter-opencv.git
cd finger-counter-opencv
```

2. Run the script:

```bash
python finger_counter.py
```

3. Place your hand inside the red rectangle on screen. Wait for ~2–3 seconds while the background is captured. Then, start showing your fingers!

Press `Esc` to exit.

---

## 📁 File Structure

```
finger-counter-opencv/
│
├── finger_counter.py       # Main application
├── README.md               # Project documentation
└── requirements.txt        # (Optional) Python dependencies
```

---

## 📸 Screenshots

<!-- Add your screenshots or demo GIF here -->
<!-- Example: -->
<!-- ![Demo](demo.gif) -->

---

## 🔧 To-Do & Improvements

- Add GUI buttons or hand gesture control
- Save finger count to a log or file
- Gesture classification model for more complex commands
- Make it mobile/cross-platform compatible

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Credits

Created with 💻 using Python and OpenCV.  
Inspired by tutorials and open-source contributions from the computer vision community.

---

## 🤝 Contributing

Pull requests, feature suggestions, and issues are welcome! Let's build this better together. ✨
```
