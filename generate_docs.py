
from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        # Logo
        # self.image('logo.png', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'AI Nutrition Planner - Project Documentation', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, num, label):
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, body):
        # Read text file
        self.set_font('Times', '', 12)
        # Output justified text
        self.multi_cell(0, 5, body)
        # Line break
        self.ln()

def create_pdf(filename):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Title Page
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 60, '', 0, 1) # vertical spacer
    pdf.cell(0, 10, 'AI Nutrition Planner', 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Comprehensive Project Explanation', 0, 1, 'C')
    pdf.cell(0, 10, f'Generated on {datetime.date.today()}', 0, 1, 'C')
    pdf.add_page()

    # SECTION 1: INTRODUCTION
    pdf.chapter_title(1, 'Introduction')
    pdf.chapter_body(
"""The AI Nutrition Planner is an intelligent, full-stack web application designed to help users manage their dietary health through advanced AI technologies. Unlike simple calorie counters, this system uses Computer Vision (OCR) to read medical reports, Generative AI (LLMs) to create personalized recipes, and comprehensive analytics to track progress.

The goal is to bridge the gap between medical data (from lab reports) and daily actionable habits (eating correct meals).""")

    # SECTION 2: KEY FEATURES
    pdf.chapter_title(2, 'Key Features')
    pdf.chapter_body(
"""1. Medical Report Analysis (OCR + AI)
   - Users can upload PDF or Image based medical reports.
   - The system uses a multi-stage parser (pypdf, PaddleOCR, Tesseract) to extract text.
   - Generative AI (Gemma/Ollama) analyzes the text to identify conditions (e.g., Diabetes), allergens (e.g., Peanuts), and vitals (Glucose, Cholesterol).

2. AI Recipe Generator
   - Based on the user's health profile and available ingredients.
   - Uses a local LLM to generate safe, customized recipes that strictly adhere to dietary restrictions.

3. Intelligent Food Logging
   - Users can scan food items or log typical meals.
   - The system calculates nutrition (Calories, Protein, Carbs, Fats) and flags potential conflicts with medical conditions.

4. Interactive Dashboard
   - Visualizes daily nutrition intake against targets.
   - Shows bio-metrics (BMI, metabolic age) and medical vitals.
   - Provide AI-driven tips based on daily habits.

5. Secure User System
   - Full authentication (Login/Register).
   - Data isolation ensures users only see their own private health data.""")

    # SECTION 3: TECHNICAL ARCHITECTURE
    pdf.chapter_title(3, 'Technical Stack')
    pdf.chapter_body(
"""Backend:
- Language: Python 3.9+
- Framework: FastAPI (High-performance web API)
- Database: SQLite (Lightweight relational DB) / JSON (Nutrition Registry)
- AI/ML: 
    - Ollama (Local LLM runner for Gemma:2b)
    - PaddleOCR (Advanced Optical Character Recognition)
    - Pytesseract (Backup OCR)
    - scikit-learn (Basic analytics)

Frontend:
- Logic: Vanilla JavaScript (ES6+)
- Styling: Custom CSS3 (Glassmorphism design, Responsive)
- Structure: HTML5
- Charts: Chart.js (Visualization)

Infrastructure:
- Docker: Containerization for easy deployment.
- Docker Compose: Orchestration of App + Ollama services.""")

    # SECTION 4: PROJECT STRUCTURE
    pdf.chapter_title(4, 'Project Structure')
    pdf.chapter_body(
"""/src
  /auth       - User authentication and profile management
  /ocr        - Parsing engines for PDFs and Images
  /services   - LLM integration, Rule engines, Analytics
  /main.py    - API Gateway and Endpoint definitions
  
/static
  /dashboard.html - Main user interface
  /upload.html    - Medical report upload
  /app.html       - Recipe and Food tools
  /styles.css     - Global theming

/data         - Nutrition databases (CSV/JSON)""")

    # SECTION 5: HOW IT WORKS (FLOW)
    pdf.chapter_title(5, 'User Workflow')
    pdf.chapter_body(
"""1. Onboarding: User registers and fills basic bio-metrics (Height, Weight, Age).
2. Assessment: User uploads a medical report (Blood test).
3. Analysis: System parses the report, extracts "High Cholesterol" and "Gluten intolerance".
4. Planning: User goes to "Recipe Generator", inputs "Chicken, Rice".
5. Generation: AI generates a "Gluten-Free Chicken Rice" recipe, low in cholesterol-raising fats.
6. Tracking: User logs the meal; dashboard updates daily calorie/macro rings.""")

    pdf.output(filename)
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    create_pdf("AI_Nutrition_Planner_Documentation.pdf")
