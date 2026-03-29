import json, io
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf

# ── Config (must match train.py) ──────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MFCC      = 40
HOP_LENGTH  = 512
MODEL_PATH  = "model.keras"
LABELS_PATH = "labels.json"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Recognition",
    page_icon="🎤",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = tf.keras.models.load_model(MODEL_PATH)
        labels = json.load(open(LABELS_PATH))
        return model, labels
    except Exception as e:
        return None, None

# ── Audio helpers ─────────────────────────────────────────────────────────────
def load_audio(file_bytes):
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=SAMPLE_RATE, duration=1.0)
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    return audio[:SAMPLE_RATE].astype(np.float32)

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    return mfcc[..., np.newaxis][np.newaxis]   # (1, 40, T, 1)

def predict(audio, model, labels):
    probs  = model.predict(extract_mfcc(audio), verbose=0)[0]
    idx    = int(np.argmax(probs))
    return labels[idx], float(probs[idx]), probs

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_waveform(audio):
    fig, ax = plt.subplots(figsize=(9, 2.5))
    t = np.linspace(0, 1, len(audio))
    ax.plot(t, audio, lw=0.7, color="#4f8ef7")
    ax.fill_between(t, audio, alpha=0.15, color="#4f8ef7")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

def plot_spectrogram(audio):
    fig, ax = plt.subplots(figsize=(9, 3))
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=64
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-Spectrogram")
    fig.tight_layout()
    return fig

def plot_bar(labels, probs):
    max_p  = max(probs)
    colors = ["#f7724f" if p == max_p else "#4f8ef7" for p in probs]
    fig = go.Figure(go.Bar(
        x=labels,
        y=probs,
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside"
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 1.15], title="Confidence"),
        xaxis_title="Word",
        title="Prediction Probabilities",
        height=380,
        margin=dict(t=50, b=10)
    )
    return fig

# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    model, labels = load_model()

    # Sidebar
    with st.sidebar:
        st.title("🎤 Speech Recognition")
        st.markdown("---")
        if model:
            st.success("Model loaded")
            st.markdown("**Recognises these words:**")
            for w in labels:
                st.markdown(f"• `{w}`")
        else:
            st.error("No model found.\nRun `python train.py` first.")
        st.markdown("---")
        st.caption("Speak clearly, ~1 second per word")

    st.title("🎤 Speech Word Recognition")
    st.caption("CNN model trained on Google Speech Commands · 10 words")

    if model is None:
        st.error("⚠️ Model not loaded. Please run `python train.py` first.")
        return

    # ── Input tabs: Upload OR Microphone ─────────────────────────────────────
    tab_upload, tab_mic = st.tabs(["Upload File", "Record Microphone"])

    audio_bytes = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a .wav / .mp3 / .flac file",
            type=["wav", "mp3", "flac", "m4a"]
        )
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes)

    with tab_mic:
        st.info("Click the mic button, say one of the 10 words clearly, then stop.")
        recorded = st.audio_input("Record your voice")   # built-in Streamlit widget
        if recorded:
            audio_bytes = recorded.read()
            st.audio(audio_bytes)

    # ── Process ───────────────────────────────────────────────────────────────
    if audio_bytes is None:
        st.markdown("---")
        st.info("Upload a file or record your voice to get started.")
        return

    with st.spinner("Analysing…"):
        try:
            audio = load_audio(audio_bytes)
        except Exception as e:
            st.error(f"Could not read audio: {e}")
            return
        word, confidence, probs = predict(audio, model, labels)

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Results")

    col1, col2 = st.columns([1, 2])
    with col1:
        if confidence >= 0.6:
            st.success(f"### {word.upper()}")
        else:
            st.warning(f"### {word.upper()}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Duration",   "1.0 s")
        st.metric("Sample Rate", f"{SAMPLE_RATE} Hz")

    with col2:
        st.plotly_chart(plot_bar(labels, probs.tolist()), use_container_width=True)

    # ── Visualisations ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Audio Visualisations")
    vtab1, vtab2 = st.tabs(["Waveform", "Spectrogram"])

    with vtab1:
        fig = plot_waveform(audio)
        st.pyplot(fig)
        plt.close(fig)

    with vtab2:
        fig = plot_spectrogram(audio)
        st.pyplot(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()