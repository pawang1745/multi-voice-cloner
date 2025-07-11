# 🎙️ Multi-Voice Cloner

A web application that allows users to **clone voices using multiple models** (.pth) on uploaded audio files. Built with **Flask**, integrated with **RVC (Retrieval-Based Voice Conversion)** for backend inference.  
This app splits the audio into chunks and clones each chunk using a different model, giving multi-style voice cloning capability.

---
## 📦 Features

- 🎧 Upload a `.wav` audio file
- 🎯 Upload **multiple voice cloning models** (`.pth`)
- 🧠 Automatically splits audio based on model count
- 🤖 Clones each chunk using RVC inference
- 📥 Downloads generated `.wav` files
- 🌐 Simple and responsive web UI with background image

---
