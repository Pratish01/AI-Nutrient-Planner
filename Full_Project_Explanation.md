# AI Nutrition Planner - System Documentation

**Date:** January 26, 2026  
**Version:** 1.0

---

## 1. Executive Summary
The **AI Nutrition Planner** is an intelligent health management system designed to bridge the gap between medical diagnostics and daily dietary habits. Unlike standard calorie counters, this application uses **Computer Vision** to read actual medical reports and **Generative AI** to create medically-safe, personalized recipes.

**Core Mission:** To empower users with chronic conditions (e.g., Diabetes, Hypertension) to eat safely without needing to be nutrition experts.

---

## 2. System Architecture & Workflow
The application operates through a strict **4-Phase Data Flow**:

### Phase 1: Onboarding & Biometrics
- **Input:** User registers and provides `Age`, `Height`, `Weight`, `Gender`.
- **Process:**
  - System calculates **BMR** (Basal Metabolic Rate) and **TDEE** (Total Daily Energy Expenditure).
  - This establishes the "Baseline Caloric Budget" for the user.
- **Output:** A user profile with daily calorie/macro targets.

### Phase 2: Medical Intelligence (The "Brain")
- **Input:** User uploads a medical report (PDF or Image).
- **Process:**
  1. **OCR Layer:** The system uses **PaddleOCR** (primary) or **Tesseract** (fallback) to extract raw text from the file. It handles both digital PDFs and scanned images.
  2. **AI Analysis:** The extracted text is fed into a **Local LLM (Gemma:2b via Ollama)**.
  3. **Extraction:** The AI parses the text to identify:
     - **Conditions:** (e.g., "Type 2 Diabetes", "Hypertension")
     - **Allergens:** (e.g., "Peanuts", "Gluten")
     - **Vitals:** (e.g., "Fasting Glucose: 140 mg/dL")
- **Output:** A **Medical Profile** stored in the database. This profile acts as a permanent filter for all future interactions.

### Phase 3: Smart Contextual Tracking
- **Input:** User scans a food item (e.g., "Cheeseburger").
- **Safety Engine:** The system performs a **Cross-Reference Check**:
  - *Retrieves:* Nutritional content of the food.
  - *Checks against:* User's Medical Profile.
  - *Logic:* "Does this food violate any medical constraints?" (e.g., Is Sodium > Limit for Hypertension?).
- **Result:** Real-time feedback is provided ("High Risk", "Moderate", "Safe") along with a visual breakdown.

### Phase 4: Generative Recipe Engine
- **Input:** User requests a meal using specific ingredients (e.g., "Chicken, Spinach").
- **Process:** The AI generates a unique recipe by solving for:
  ```text
  Result = [User Ingredients] 
           INTERSECT [Medical Constraints from Phase 2] 
           INTERSECT [Caloric Budget from Phase 1]
  ```
- **Output:** A tailored recipe (e.g., "Low-Sodium Lemon Garlic Chicken") with exact macro-nutrients.

---

## 3. Technology Stack

### Backend Infrastructure
- **Python 3.9+:** The core logic language, chosen for its dominance in AI/ML.
- **FastAPI:** A modern, high-performance web framework for building APIs. It handles the "glue" between the user, the database, and the AI models.
- **SQLite:** A serverless, self-contained SQL database engine. Used for storing User Profiles and Medical Data securely locally.
- **Docker:** Used to containerize the application, ensuring it runs identically on any machine (Windows/Linux/Mac).

### Artificial Intelligence Suite
- **Ollama:** A local runtime for running Large Language Models off-line. It ensures user health data stays private on the device.
- **Gemma:2b:** Google's lightweight open model. Used for text reasoning and recipe generation.
- **PaddleOCR:** A deep-learning based Optical Character Recognition tool. It provides superior accuracy for reading scanned documents compared to traditional tools.
- **Scikit-learn:** Used for basic analytical regressions and data processing.

### Frontend Interface
- **HTML5 / CSS3 / JavaScript:** Built using "Vanilla" technologies (no heavy frameworks like React) to ensure maximum speed and simplicity.
- **Glassmorphism Design:** The UI features a modern, translucent aesthetic (frosted glass effect) utilizing CSS backdrop-filters.
- **Chart.js:** A JavaScript library used to render the dynamic "Donut" charts for nutrition tracking and "Bar" charts for progress.

---

## 4. Project Structure (Key Files)

```
/AI Nutrition
├── /src
│   ├── main.py              # The "Brain" - Entry point for the API
│   ├── /auth                # Handles User Login/Register security
│   ├── /ocr                 # Scripts for reading PDF/Images (PaddleOCR)
│   ├── /services            # Connects to LLM (Ollama) and Logic Engines
│   └── /rules               # Hard-coded dietary rules (e.g., "Sugar is bad for Diabetes")
├── /static
│   ├── dashboard.html       # Main user dashboard interface
│   ├── upload.html          # Interface for uploading medical reports
│   ├── app.html             # Recipe Generator & Food Scanner tool
│   └── styles.css           # Global definitions for colors/fonts
├── /data                    # Database files (nutrition.db)
└── docker-compose.yml       # Instructions to run the whole app
```

---

## 5. How to Run
1. Ensure **Docker** and **Ollama** are installed.
2. Open a terminal in the project folder.
3. Run: `docker-compose up --build`
4. Open your browser to: `http://localhost:8000`
