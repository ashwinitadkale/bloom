# bloom

# 🌸 AI-Powered Menstrual Cycle Prediction System

An end-to-end **Machine Learning–powered web application** that predicts the number of days until the next menstrual cycle using user-provided health and cycle information.  
This project demonstrates **ML pipeline design, backend integration using Flask, and a responsive HTML dashboard**.

> ⚠️ **Note:** This project is a prototype built for learning and portfolio purposes, not a medical or clinical tool.

---

## 🚀 Project Overview

Menstrual health data is sensitive and often unavailable due to privacy and ethical constraints.  
To address this, this project uses **synthetic data** to simulate realistic menstrual cycle patterns and focuses on building a **complete ML system**, rather than clinical accuracy.

The system allows users to:
- Enter basic cycle and wellness information
- Get a prediction for the next cycle (in days)
- View results through an interactive dashboard UI

---

## 🧠 How It Works (High Level)

1. **Synthetic dataset** is generated to reflect realistic menstrual cycle patterns  
2. A **baseline regression model** is trained to predict days until the next cycle  
3. The trained model is saved using `pickle`  
4. A **Flask backend API** loads the model and serves predictions  
5. An **HTML + Tailwind CSS frontend** collects user inputs and displays results  

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- NumPy
- Pandas
- Scikit-learn

### Backend
- Flask
- Pickle (model serialization)

### Frontend
- HTML
- Tailwind CSS
- JavaScript
- Chart.js (static visualization)
- FullCalendar.js (static calendar UI)

---

## 📊 Dataset Information

- The dataset used is **synthetically generated**
- No real user or medical data is used
- Data generation is based on:
  - Typical menstrual cycle ranges (21–35 days)
  - Period duration (3–7 days)
  - Common symptoms and mood patterns

### Why Synthetic Data?
- Menstrual health data is highly sensitive
- Avoids ethical and privacy issues
- Allows safe demonstration of ML workflows

This choice is **intentional and documented**.

---

## 🧮 Model Details

- **Type:** Regression model (baseline)
- **Target Variable:** Days until next menstrual cycle
- **Input Features:**
  - Age
  - Average cycle length
  - Days since last period
  - Mood (encoded)
  - Flow intensity (encoded)
  - Primary symptom (encoded)

> The model is designed as a **proof-of-concept**, not for clinical reliability.

---

## 🖥️ Application Features

- ML-powered prediction endpoint (`/predict`)
- Flask-based backend integration
- Interactive HTML dashboard
- User-friendly input form
- Modular and extensible architecture

---

## 📂 Project Structure

