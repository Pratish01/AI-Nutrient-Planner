
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import datetime

def create_detailed_report(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    Story = []
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, fontSize=11, leading=14))
    styles.add(ParagraphStyle(name='TechTitle', parent=styles['Heading3'], textColor=colors.darkblue))
    
    # --- TITLE PAGE ---
    Story.append(Spacer(1, 100))
    Story.append(Paragraph("AI Nutrition Planner", styles["Title"]))
    Story.append(Paragraph("System Architecture & Workflow", styles["Heading2"]))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(f"Generated on {datetime.date.today()}", styles["Normal"]))
    Story.append(PageBreak())

    # --- SECTION 1: SYSTEM FLOW ---
    Story.append(Paragraph("1. Detailed System Workflow", styles["Heading1"]))
    Story.append(Paragraph("The application operates through four distinct, interconnected phases:", styles["Normal"]))
    Story.append(Spacer(1, 12))

    # Phase 1
    Story.append(Paragraph("Phase 1: User Onboarding & Profile Creation", styles["Heading2"]))
    Story.append(Paragraph("""
    <b>Input:</b> User Registration (Email/Password) + Bio-metrics (Age, Height, Weight).<br/>
    <b>Process:</b>
    1. <b>Auth Service:</b> Hashes password (bcrypt) and generates JWT token for session management.
    2. <b>Calculation Engine:</b> Computes BMR (Basal Metabolic Rate) and TDEE (Total Daily Energy Expenditure) to establish baseline caloric needs.<br/>
    <b>Output:</b> Encrypted User Record + Baseline Nutrition Targets.
    """, styles["Justify"]))

    # Phase 2
    Story.append(Paragraph("Phase 2: Medical Intelligence (The Core)", styles["Heading2"]))
    Story.append(Paragraph("""
    <b>Input:</b> Medical Report (PDF or Image format).<br/>
    <b>Process:</b>
    1. <b>Upload Handler:</b> API Endpoint receives file, saves to temp storage.
    2. <b>OCR Pipeline (Multi-Strategy):</b>
       - <i>Strategy A:</i> Digital text extraction via <code>pypdf</code> (Fastest).
       - <i>Strategy B:</i> Embedded image extraction via <code>pypdf</code> + <code>PaddleOCR</code> (For scanned PDFs).
       - <i>Strategy C:</i> Full page rendering via <code>pdf2image</code> + <code>Tesseract</code> (Fallback).
    3. <b>AI Analysis:</b> Extracted raw text is sent to the LLM Service (<b>Ollama/Gemma</b>).
    4. <b>Prompt Engineering:</b> A specialized "Medical Parser" prompt instructs the AI to extract strictly structured JSON data: Conditions (e.g., "Type 2 Diabetes"), Allergens, and Vitals.<br/>
    <b>Output:</b> Structured "Medical Profile" stored in the database.
    """, styles["Justify"]))
    
    # Phase 3
    Story.append(Paragraph("Phase 3: Context-Aware Food Tracking", styles["Heading2"]))
    Story.append(Paragraph("""
    <b>Input:</b> Food scan (Image) or Text Search.<br/>
    <b>Process:</b>
    1. <b>Retrieval:</b> Fetches nutrition data (Calories, Macros) for the item.
    2. <b>Safety Engine:</b> Cross-references the food's ingredients against the user's Medical Profile.
       - <i>Example:</i> If User has "Hypertension", engine flags "High Sodium" foods.
    3. <b>Aggregator:</b> Updates daily totals against the TDEE limits calculated in Phase 1.<br/>
    <b>Output:</b> Real-time feedback ("Safe", "Warning") and ring-chart updates.
    """, styles["Justify"]))

    # Phase 4
    Story.append(Paragraph("Phase 4: Generative Recipe Creation", styles["Heading2"]))
    Story.append(Paragraph("""
    <b>Input:</b> User Ingredients List (e.g., "Chicken, Onion").<br/>
    <b>Process:</b>
    1. <b>Context Assembly:</b> System combines [Ingredients] + [Medical Conditions] + [Allergens].
    2. <b>Generative Request:</b> Sends prompt to LLM: "Create a recipe using [Ingredients] that is safe for [Conditions]".
    3. <b>JSON Enforcement:</b> LLM returns recipe in strict JSON format for UI rendering.<br/>
    <b>Output:</b> Personalized, medically-safe recipe card with instructions and estimated nutrition.
    """, styles["Justify"]))
    
    Story.append(PageBreak())

    # --- SECTION 2: TECHNOLOGY STACK ---
    Story.append(Paragraph("2. Technology Stack & Rationale", styles["Heading1"]))

    # Backend
    Story.append(Paragraph("Backend Infrastructure", styles["Heading2"]))
    data = [
        ["Component", "Technology", "Role & Rationale"],
        ["Language", "Python 3.9+", "Dominant in AI/ML ecosystem, huge library support."],
        ["Framework", "FastAPI", "Async, high-performance, auto-documentation (Swagger UI)."],
        ["Database", "SQLite", "Serverless, zero-configuration, ACID compliant. Perfect for embedded/local apps."],
        ["Container", "Docker", "Ensures environment consistency across dev/prod machines."]
    ]
    t = Table(data, colWidths=[80, 100, 320])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.navy),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
    ]))
    Story.append(t)
    Story.append(Spacer(1, 12))

    # AI
    Story.append(Paragraph("Artificial Intelligence Suite", styles["Heading2"]))
    data_ai = [
        ["Module", "Tech Stack", "Role"],
        ["LLM Runtime", "Ollama", "Runs large language models locally (offline privacy)."],
        ["Model", "Google Gemma:2b", "Efficient, lightweight open-weights model capable of reasoning."],
        ["Primary OCR", "PaddleOCR", "State-of-the-art deep learning OCR. Handles angles/blur."],
        ["Backup OCR", "Tesseract / pypdf", "Robust fallback for digital PDFs or simple images."]
    ]
    t2 = Table(data_ai, colWidths=[80, 120, 300])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
    ]))
    Story.append(t2)
    Story.append(Spacer(1, 12))

    # Frontend
    Story.append(Paragraph("Frontend Interface", styles["Heading2"]))
    Story.append(Paragraph("""
    <b>Architecture:</b> Multi-page Static App (SPA-feel).<br/>
    <b>Core:</b> HTML5, Native ES6 JavaScript (No build step required).<br/>
    <b>Styling:</b> CSS3 Custom Properties (Variables) for easy theming (Dark/Light modes).<br/>
    <b>Design System:</b> "Glassmorphism" - Semi-transparent cards with backdrop-blur filters for a modern, premium aesthetic.<br/>
    <b>Visualization:</b> <code>Chart.js</code> for rendering dynamic nutrition donuts and bar charts.
    """, styles["Justify"]))

    doc.build(Story)
    print(f"Detailed Architecture PDF Generated at: {filename}")

if __name__ == "__main__":
    create_detailed_report("AI_Nutrition_System_Architecture_v3.pdf")
