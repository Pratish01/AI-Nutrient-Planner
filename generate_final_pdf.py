
import sys
import os

# Ensure fpdf is importable
try:
    from fpdf import FPDF
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'AI Nutrition Planner - Complete Documentation', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 5, text)
        self.ln()

def generate():
    pdf = PDF()
    pdf.add_page()
    
    # CONTENT
    
    # 1. Overview
    pdf.chapter_title("1. Executive Summary")
    pdf.chapter_body(
        "The AI Nutrition Planner is a comprehensive health management system designed to bridge the gap between "
        "medical diagnostics and daily nutrition. By integrating Computer Vision (OCR) and Generative AI (LLMs), "
        "the system analyzes complex medical reports to create personalized, medically-safe meal plans.\n\n"
        "Core Value Proposition:\n"
        "- Automates the understanding of medical restrictions (Diabetes, Allergies).\n"
        "- Generates custom recipes that strictly adhere to these restrictions.\n"
        "- Provides a privacy-first, local-AI solution for health data."
    )

    # 2. Architecture
    pdf.chapter_title("2. System Architecture & Workflow")
    pdf.chapter_body(
        "The system follows a linear 4-phase data flow:\n\n"
        "Phase 1: Onboarding & Biometrics\n"
        "Users register and input bio-metrics (Age, Height, Weight). The system calculates BMR (Basal Metabolic Rate) "
        "and TDEE (Total Daily Energy Expenditure) to establish baseline caloric limits.\n\n"
        "Phase 2: Medical Intelligence (The Core)\n"
        "Input: PDF/Image Medical Report.\n"
        "- OCR Layer: PaddleOCR extracts raw text from digital or scanned documents.\n"
        "- AI Layer: Ollama (Gemma:2b) analyzes the text to extract 'Conditions' (e.g., Hypertension), 'Allergens', and 'Vitals'.\n"
        "Output: A 'Medical Profile' stored in the database.\n\n"
        "Phase 3: Smart Contextual Tracking\n"
        "When a user scans a food item, the system Cross-References the food's nutritional content against the user's "
        "Medical Profile. It flags dangerous interactions (e.g., High Sodium food for Hypertensive users).\n\n"
        "Phase 4: Generative Recipe Engine\n"
        "The user requests a recipe using available ingredients. The AI generates a result that satisfies:\n"
        "[Ingredients] + [Medical Constraints] + [Taste Preferences]."
    )

    # 3. Technologies
    pdf.chapter_title("3. Technology Stack")
    pdf.chapter_body(
        "Backend Infrastructure:\n"
        "- Python 3.9+: Core logic implementation.\n"
        "- FastAPI: High-performance asynchronous API framework.\n"
        "- SQLite: Serverless relational database for user data.\n"
        "- Docker: Containerization for reproducible deployments.\n\n"
        "Artificial Intelligence:\n"
        "- Ollama: Local runtime for Large Language Models (LLMs).\n"
        "- Gemma:2b: Google's lightweight open model for text reasoning.\n"
        "- PaddleOCR: Deep Learning based Optical Character Recognition.\n"
        "- Tesseract: Fallback OCR engine.\n\n"
        "Frontend Interface:\n"
        "- Architecture: Multi-page Application serving static HTML.\n"
        "- Tech: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript.\n"
        "- Visualization: Chart.js for nutrition analytics."
    )

    # 4. Folder Structure
    pdf.chapter_title("4. Project Structure Checklist")
    pdf.chapter_body(
        "/src\n"
        "  /auth       - JWT Authentication & User Management\n"
        "  /ocr        - PDF Parsing & Image Processing Strategies\n"
        "  /services   - LLM Integration & Rule Engines\n"
        "  /main.py    - API Gateway Entry point\n\n"
        "/static\n"
        "  /dashboard.html - Main User Interface\n"
        "  /upload.html    - File Upload Interface\n"
        "  style.css       - Global Design Token definitions\n\n"
        "/data         - Persistence layer (DBs and Registries)"
    )

    output_path = "Full_Project_Documentation.pdf"
    pdf.output(output_path)
    print(f"SUCCESS: Generated {os.path.abspath(output_path)}")

if __name__ == "__main__":
    try:
        generate()
    except Exception as e:
        print(f"ERROR: {e}")
