
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import datetime

def create_report(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    Story = []
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))

    # TITLE PAGE
    title = "AI Nutrition Planner"
    subtitle = "Project Documentation & Technical Overview"
    date = f"Generated on: {datetime.date.today()}"
    
    Story.append(Spacer(1, 60))
    Story.append(Paragraph(title, styles["Title"]))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(subtitle, styles["Heading2"]))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(date, styles["Normal"]))
    Story.append(PageBreak())

    # INTRODUCTION
    Story.append(Paragraph("1. Introduction", styles["Heading1"]))
    intro_text = """
    The AI Nutrition Planner is an advanced web application that merges Computer Vision, Generative AI, and traditional web technologies to provide personalized health management.
    
    It addresses the complexity of modern diet management by automating the analysis of medical data (blood reports) and converting that data into actionable, safe meal plans and recipes.
    """
    Story.append(Paragraph(intro_text, styles["Justify"]))
    Story.append(Spacer(1, 12))

    # FEATURES
    Story.append(Paragraph("2. Key Features", styles["Heading1"]))
    
    features = [
        ["Feature", "Description"],
        ["Medical OCR", "Extracts diagnosis/vitals from PDF/Images using PaddleOCR & AI."],
        ["AI Parsing", "Uses LLM (Gemma) to understand complex medical text structure."],
        ["Recipe Engine", "Generates safe recipes using Generative AI based on health profile."],
        ["Food Logging", "Visual dashboard for tracking daily intake against medical limits."],
        ["Privacy", "Strict data isolation ensuring user data remains private."]
    ]
    
    t = Table(features, colWidths=[100, 340])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    Story.append(t)
    Story.append(Spacer(1, 24))

    # TECH STACK
    Story.append(Paragraph("3. Technical Architecture", styles["Heading1"]))
    tech_text = """
    <b>Backend:</b> Python (FastAPI)<br/>
    <b>AI Services:</b> Ollama (LLM), PaddleOCR (Vision), NumPy<br/>
    <b>Frontend:</b> HTML5, CSS3, JavaScript (Vanilla)<br/>
    <b>Database:</b> SQLite (Performance) & JSON (Registry)<br/>
    <b>Deployment:</b> Docker & Docker Compose
    """
    Story.append(Paragraph(tech_text, styles["Normal"]))
    Story.append(Spacer(1, 12))

    # FOLDER STRUCTURE
    Story.append(Paragraph("4. Project Structure", styles["Heading1"]))
    struct_data = [
        ["Directory", "Purpose"],
        ["/src/main.py", "Entry point, API routes, and App logic."],
        ["/src/ocr/", "OCR engines (Parser script & strategies)."],
        ["/src/services/", "LLM integration and Business logic."],
        ["/static/", "Frontend assets (HTML, CSS, JS)."],
        ["/data/", "Database files and CSV datasets."]
    ]
    t2 = Table(struct_data, colWidths=[100, 340])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    Story.append(t2)

    # GENERATION
    doc.build(Story)
    print(f"Successfully generated: {filename}")

if __name__ == "__main__":
    create_report("Project_Report_v2.pdf")
