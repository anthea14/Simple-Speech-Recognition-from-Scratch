# Simple-Speech-Recognition-from-Scratch

A deep learning-based speech recognition system that identifies spoken words in real-time, built with TensorFlow and deployed via a Streamlit web interface.

---

## Demo

Upload a `.wav`, `.mp3`, or `.flac` audio file — or record directly from your microphone — and the system will predict the spoken word with a confidence score.

---

## Features

- 🎤 **Live microphone recording** via browser
- 📁 **Audio file upload** (WAV, MP3, FLAC, M4A)
- 📊 **Confidence score** with probability chart for all 10 classes
- 🧠 **CNN model** trained on Google Speech Commands dataset
- ⚡ **Fast inference** — results in under a second

---

## Recognisable Words

The model is trained to recognise these 10 words:

| | | | | |
|---|---|---|---|---|
| yes | no | up | down | left |
| right | on | off | stop | go |

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | TensorFlow / Keras (CNN) |
| Audio Processing | Librosa |
| Web Interface | Streamlit |
| Charts | Plotly |
| Dataset | Google Speech Commands V2 |
| Language | Python 3.11 |

---

## Project Structure
```
speech-project/
├── app.py              # Streamlit web application
├── model.keras         # Trained CNN model
├── labels.json         # Class label mappings
└── requirements.txt    # Python dependencies
```

---

## Model Architecture
```
Input (MFCC features)
        ↓
Conv2D (32 filters) + BatchNorm + MaxPool + Dropout
        ↓
Conv2D (64 filters) + BatchNorm + MaxPool + Dropout
        ↓
Conv2D (128 filters) + BatchNorm + GlobalAveragePool + Dropout
        ↓
Dense (128) + Dropout
        ↓
Softmax Output (10 classes)
```

- **Input features:** MFCC (Mel-Frequency Cepstral Coefficients)
- **Training data:** ~30,000 audio samples, 1 second each at 16kHz
- **Validation accuracy:** ~85–90%

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOURUSERNAME/speech-project.git
cd speech-project
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How It Works

1. **Audio Input** — User uploads a file or records via microphone
2. **Preprocessing** — Audio is resampled to 16kHz and trimmed/padded to 1 second
3. **Feature Extraction** — MFCC features are extracted using Librosa
4. **Inference** — The CNN model predicts probabilities for all 10 word classes
5. **Output** — The top prediction is displayed with confidence score and probability bar chart

---

## Dataset

Trained on the **Google Speech Commands V2** dataset:
- 105,000+ audio clips
- 35 word categories
- 1 second per clip at 16kHz
- This project uses 10 of the most common command words

---

## Requirements
```
tensorflow
librosa
streamlit
plotly
matplotlib
```

---
