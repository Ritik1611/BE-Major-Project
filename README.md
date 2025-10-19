# 🧠 Privacy-Preserving AI Framework

**Local Data Agent (LDA) + Trainer Agent + DP Agent + Encryption Agent + Orchestrator**

> A modular privacy-preserving AI framework enabling client-side preprocessing, secure encrypted storage, federated/DP training, and explainable evaluation — designed for depression/anxiety (MDD, GAD, PTSD) detection through multimodal (text, audio, video) data.

---

## 🏗️ Project Overview

### **Architecture Summary**

The system is divided into **Client-side Agents** and **Server-side Agents**:

| Layer           | Components                    | Description                                                                                                |
| --------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Client Side** | 🧩 **Local Data Agent (LDA)** | Handles data ingestion and local multimodal preprocessing (text/audio/video) with encryption and receipts. |
|                 | 🧠 **Trainer Agent**          | Fine-tunes and evaluates models (MentalBERT, local probe) securely.                                        |
|                 | 🔐 **DP Agent**               | Applies differential privacy noise mechanisms (Gaussian, Laplace, Uniform, Exponential, Student-t).        |
|                 | 🔒 **Encryption Agent**       | Manages AES-GCM, Fernet, KMS-envelope, and Homomorphic Encryption (CKKS) modes.                            |
| **Server Side** | ⚙️ **Orchestrator Agent**     | Coordinates client-side training, DP application, explainability, and logging.                             |
|                 | 🧾 **Aggregator Agent**       | Aggregates deltas (model updates) securely using secure aggregation.                                       |
|                 | 🔑 **Key Management Agent**   | Issues and rotates cryptographic keys (used by SecureStore).                                               |
|                 | 🧩 **Audit Agent**            | Logs and monitors training, DP application, and explainability integrity for compliance.                   |

---

## 📦 Repository Layout

```
BE-Major-Project/
├── LDA/
│   ├── app/
│   │   ├── main.py                     # FastAPI entrypoint for local preprocessing
│   │   ├── pipelines/
│   │   │   ├── text.py                 # Text preprocessing (PII scrub, embeddings)
│   │   │   ├── audio.py                # Audio feature extraction (wav2vec2, openSMILE, prosody)
│   │   │   ├── session_processor.py    # Session-level AV diarization, ASR, face tracking
│   │   │   └── video.py                # OpenFace-based facial analysis and blurring
│   │   ├── utils/receipts.py           # Signed receipt generator (HMAC)
│   │   └── security/secure_store.py    # AES-GCM encrypted SecureStore
│   ├── configs/local_config.yaml       # Config (paths, OpenFace binary, storage root)
│   └── secure_store/                   # Encrypted outputs (auto-created)
│
├── trainer_agent/trainer_mentalbert_privacy.py  # Trainer agent orchestrator
├── dp_agent/dp_agent.py                         # DP noise injector
├── enc_agent/enc_agent.py                       # Encryption interface (AES, Fernet, CKKS)
├── centralized_secure_store.py                   # Shared encryption logic for central nodes
├── centralized_receipts.py                       # Global receipt manager
├── create_dp_comparison.py                       # Main orchestrator (integrates all agents)
└── format_daic_to_lda.py                         # Converts DAIC-WOZ dataset to LDA format
```

---

## 🧩 Core Components

### 1️⃣ Local Data Agent (LDA)

Handles all client-side multimodal ingestion:

* **Text:** spaCy-based PII scrub + BERT embeddings
* **Audio:** wav2vec2 / openSMILE eGeMAPS / prosody extraction
* **Video:** OpenFace for facial Action Units + blurring faces for anonymization
* **Encryption:** AES-GCM (HKDF derived context key per file)
* **Receipts:** Signed with HMAC for audit verification

Command (example):

```bash
uvicorn app.main:app --reload
# or directly run pipeline:
python3 LDA/app/main.py --input-type text_dir --input-path ./samples --mode session
```

---

### 2️⃣ Trainer Agent

Fine-tunes **MentalBERT** or fallback **probe model** for local mode training.

Key features:

* Supports *binarized PHQ thresholding*
* Works with *MentalBERT*, fallback *AutoModel*, or internal *probe model*
* Produces explainability logs (feature importance per epoch)
* Generates `local_probe_*.pt` weights for DP stage

```bash
python3 trainer_agent/trainer_mentalbert_privacy.py --train --epochs 3 --model mental/mental-bert-base-uncased
```

---

### 3️⃣ Differential Privacy Agent

Implements noise injection and privacy evaluation.

Supported mechanisms:

* Gaussian
* Laplace
* Uniform
* Exponential
* Student-t
* None (baseline)

Metrics computed:

* Accuracy / F1 / MAE / silhouette score
* Privacy–utility tradeoff graphs

CSV and plots saved as:

```
dp_noise_mechanism_comparison_<mode>.csv
plots/<mode>_noise_vs_metric.png
```

---

### 4️⃣ Encryption Agent

Implements multiple encryption strategies:

* AES-GCM (default)
* Fernet symmetric encryption
* AWS KMS envelope encryption (mocked local)
* Homomorphic encryption (CKKS via TenSEAL)
* Secure Multi-Party Computation (SMP-C placeholder)

Integrated with `SecureStore` in both local and central storage.

---

### 5️⃣ Orchestrator (create_dp_comparison.py)

Central execution entrypoint.
Runs:

* LDA preprocessing
* Trainer orchestration
* Differential Privacy evaluation
* Explainability
* DP plots + CSV exports

---

## ⚙️ Installation Guide

### 🔹 Common Dependencies

Install system packages:

```bash
sudo apt install ffmpeg libsndfile1 python3-dev build-essential cmake git
```

Then install Python deps:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch torchvision torchaudio
transformers
pandas pyarrow numpy scipy scikit-learn
fastapi uvicorn
opencv-python ffmpeg-python librosa webrtcvad pyannote.audio
pydub matplotlib seaborn
cryptography
tenseal
openpyxl
```

---

## 🎥 Setting Up OpenFace (for Video Analysis)

### **Linux / Arch Linux**

```bash
# install dependencies
sudo pacman -S cmake dlib opencv ffmpeg
# build OpenFace
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
bash download_models.sh
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Verify:

```bash
./bin/FeatureExtraction -h
```

Update path in config:

```yaml
video_pipe:
  openface:
    binary_path: /home/<user>/Desktop/BE-Major-Project/LDA/OpenFace/build/bin/FeatureExtraction
```

---

### **Windows Setup**

1. Install [CMake](https://cmake.org/download/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Clone OpenFace:

   ```powershell
   git clone https://github.com/TadasBaltrusaitis/OpenFace.git
   cd OpenFace
   .\download_models.ps1
   mkdir build; cd build
   cmake .. -G "Visual Studio 17 2022" -A x64
   cmake --build . --config Release
   ```
3. Update `local_config.yaml`:

   ```yaml
   video_pipe:
     openface:
       binary_path: C:/Users/<username>/BE-Major-Project/LDA/OpenFace/build/bin/FeatureExtraction.exe
   ```

---

## 🎙️ Setting Up openSMILE (Audio Features)

### **Linux / Arch**

```bash
sudo pacman -S portaudio sox
git clone https://github.com/audeering/opensmile.git
cd opensmile
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### **Windows**

1. Download prebuilt openSMILE binaries from:
   [https://audeering.github.io/opensmile/download/](https://audeering.github.io/opensmile/download/)
2. Extract to `C:\Program Files\openSMILE\bin`
3. Add to PATH:

   ```
   setx PATH "%PATH%;C:\Program Files\openSMILE\bin"
   ```

Then update path in your LDA config:

```yaml
audio_pipe:
  opensmile:
    binary_path: "C:/Program Files/openSMILE/bin/SMILExtract.exe"
```

---

## 🧩 Running the Full Pipeline

Example (all-in-one orchestrator):

```bash
python3 create_dp_comparison.py \
  --lda-mode session \
  --input-type text_dir \
  --input-path ./sample_texts \
  --store-root ./secure_store \
  --modes base rag vector_rag \
  --epochs 20 \
  --lr 1e-3 \
  --use-bert \
  --device cuda \
  --rag-k 3
```

This will automatically:

1. Preprocess text/audio/video via LDA
2. Train MentalBERT (or fallback local probe)
3. Generate explainability logs under `explain_logs/`
4. Run DP noise mechanism comparison
5. Save per-mode CSVs and combined results

---

## 🧾 Output Directories

| Directory                             | Contents                    |
| ------------------------------------- | --------------------------- |
| `secure_store/`                       | Encrypted parquet, receipts |
| `trainer_outputs/`                    | Local probe `.pt` models    |
| `plots/`                              | Noise–metric graphs         |
| `explain_logs/`                       | Text explainability reports |
| `dp_noise_mechanism_comparison_*.csv` | Per-mode DP summaries       |

---

## 🧠 Explainability & RAG Features

The framework includes latency explainability for RAG:

* `rag_mean_latency`
* `rag_median_latency`
* `rag_k`

These appear automatically in your CSV results:

```
mechanism,noise_multiplier,accuracy,f1,rag_mean_latency,rag_median_latency,rag_k
gaussian,0.5,0.821,0.74,3.41,2.99,3
```

---

## 🪶 Platform-Specific Notes

| Component        | Arch Linux Path                                                                | Windows Path                                                                    |
| ---------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| OpenFace binary  | `/home/user/Desktop/BE-Major-Project/LDA/OpenFace/build/bin/FeatureExtraction` | `C:\Users\<user>\BE-Major-Project\LDA\OpenFace\build\bin\FeatureExtraction.exe` |
| openSMILE binary | `/usr/local/bin/SMILExtract`                                                   | `C:\Program Files\openSMILE\bin\SMILExtract.exe`                                |
| SecureStore root | `./secure_store`                                                               | `.\secure_store`                                                                |
| Config YAML      | `configs/local_config.yaml`                                                    | same (paths adjusted)                                                           |

---

## 🧩 Troubleshooting

| Issue                                                      | Cause                            | Fix                                                                        |
| ---------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------- |
| `utf-8 codec can’t decode byte 0x80`                       | Binary DP delta read incorrectly | Use `rb` mode or `file://` prefix                                          |
| `local variable 'delta_path' referenced before assignment` | Missing default delta path       | Fixed in latest `run_pipeline()` patch                                     |
| `MentalBERT gated repo`                                    | Model not publicly available     | Use `distilbert-base-uncased` or HF login: `huggingface-cli login`         |
| `Probe metrics all zero`                                   | No numeric features              | Ensure `embedding` column in parquet or fallback BERT embeddings generated |
| `OpenFace model not found`                                 | Wrong path in YAML               | Correct `video_pipe.openface.binary_path`                                  |

---

## 🧪 Example Config (`configs/local_config.yaml`)

```yaml
mode: session
ingest:
  video: true
  audio: true
  text: true

video_pipe:
  openface:
    binary_path: /home/user/Desktop/BE-Major-Project/LDA/OpenFace/build/bin/FeatureExtraction
    classifiers_path: /home/user/Desktop/BE-Major-Project/LDA/OpenFace/build/bin/classifiers/haarcascade_frontalface_alt.xml

audio_pipe:
  opensmile:
    binary_path: /usr/local/bin/SMILExtract
  wav2vec2:
    model: facebook/wav2vec2-base-960h

storage:
  root: ./secure_store
```

---

## 🧩 Contributors

* **Ritik Shetty** – Architect & Lead Developer
* **Dr. Nupur Giri** – Project Supervisor
* **Ascentech Collaboration** – Data & Compute Resources

---

## 📘 Citation

If you use this framework in research or development:

```
@software{privacy_preserving_ai_2025,
  author = {Shetty, Ritik and Giri, Nupur},
  title = {Privacy-Preserving AI Framework for Multimodal Depression Detection},
  year = {2025},
  url = {https://github.com/ritikshetty/BE-Major-Project}
}
```

---
