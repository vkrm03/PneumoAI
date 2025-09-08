# 🩻 PneumoAI – Pneumonia Detection

PneumoAI is a **Flask-based web app** that uses a deep learning model to classify chest X-rays as **NORMAL** or **PNEUMONIA**, helping radiologists quickly identify patients at risk.

---

## 🚀 Problem Statement
Pneumonia detection through X-rays is time-consuming and prone to errors, especially in crowded hospitals.  
This project automates the process by providing instant predictions to assist doctors in early diagnosis.

---

## 📂 Dataset & Colab
- **Dataset & Colab:** [Google Drive Link](https://drive.google.com/drive/folders/1yhhzRceT4sO27BoTwfzX3VpK062I94sy?usp=drive_link)

---

## 🛠️ Setup & Run
1. **Clone the repo**
   ```bash
   git clone https://github.com/vkrm03/PneumoAI.git
   cd PneumoAI/WEB
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python app.py
   ```
   Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📸 Example
Upload a chest X-ray and get a prediction:
```
Prediction: PNEUMONIA
Confidence: 95.7%
```

---

## 📌 Notes
- For research & educational purposes only.
- Not a certified medical diagnostic tool.

---
