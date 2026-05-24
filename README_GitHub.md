# FaceSearch Bio Pro v12.3 🎯

**AUTONOMOUS EVOLUTION OSINT SUITE** — Advanced multi-modal intelligence gathering platform.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🚀 Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_app_v12_3.py
```

### Streamlit Cloud Deployment
1. Fork this repository
2. Connect your GitHub account at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New App" → Select this repo → Main file: `streamlit_app_v12_3.py`
4. Deploy!

## 📊 Capabilities (26 OSINT Modes)

| Category | Modes |
|----------|-------|
| **Image & Biometric** | Image Search, Biometric Analysis, Deep Face, Face++ Recognition, Multi-Provider Face |
| **Social & Web** | Username Enumeration, Social Media, Reddit, Telegram, Discord, YouTube OSINT |
| **Infrastructure** | Domain Analysis, Email Intelligence, Geolocation, Blockchain |
| **Dark & Threat** | Darkweb Monitor, Darknet Markets, Threat Intelligence |
| **Analysis** | Anomaly Detection, NLP Intelligence, Bot Detection, Evidence Chain |
| **Operations** | Batch Processing, Real-Time Monitor, Auto Report, Collaborative, OSINT Framework |

## 🔧 Architecture
- **48 Classes** | **175 Functions** | **5,198 Lines** | **254,529 Characters**
- **Client-Side Encryption** — Zero-knowledge architecture
- **Multi-Provider Face Recognition** — AWS/Azure/Face++/Local OpenCV with auto-fallback
- **YouTube Innertube API** — No API key required
- **Redis Caching** — Optional with in-memory fallback

## ⚠️ Legal Notice
For **educational and authorized security testing only**. Users are responsible for compliance with all applicable laws. MIT License.

## 📁 Repository Structure
```
├── streamlit_app_v12_3.py    # Main application (254KB)
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (Linux)
├── README.md                   # This file
└── .gitignore                  # Git exclusions
```

## 🔑 Optional Credentials (for enhanced features)
Configure in Streamlit Cloud Secrets (`.streamlit/secrets.toml`):
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — AWS Rekognition
- `AZURE_FACE_KEY` / `AZURE_FACE_ENDPOINT` — Azure Face API
- `FACEPP_API_KEY` / `FACEPP_API_SECRET` — Face++
- `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` — Redis Cache

---
**Version:** 12.3 | **Status:** Production Ready | **License:** MIT
