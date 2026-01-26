
import os

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Nutrition Planner - System Architecture</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 40px; background: #f9f9f9; }
        .container { background: white; padding: 50px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
        h1 { color: #00b894; border-bottom: 2px solid #00b894; padding-bottom: 10px; }
        h2 { color: #2d3436; margin-top: 30px; border-left: 5px solid #00b894; padding-left: 15px; }
        h3 { color: #636e72; margin-top: 20px; }
        .tech-box { background: #e0f2f1; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #b2dfdb; }
        .flow-step { background: #fff3e0; padding: 20px; margin: 15px 0; border-left: 5px solid #ff9800; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f1f1f1; }
        code { background: #eee; padding: 2px 5px; border-radius: 4px; font-family: 'Consolas', monospace; color: #d63031; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Nutrition Planner</h1>
        <p style="font-size: 1.2em; color: #666;">Detailed System Architecture & Workflow Documentation</p>
        <hr>

        <h2>1. System Workflow</h2>
        <p>The application follows a linear 4-phase logic flow:</p>

        <div class="flow-step">
            <h3>Phase 1: Medical Intelligence (Input)</h3>
            <p><strong>Input:</strong> User uploads a PDF/Image medical report.</p>
            <p><strong>Process:</strong></p>
            <ul>
                <li><strong>Optical Character Recognition (OCR):</strong> The system uses <code>PaddleOCR</code> (Digital/Scanned) to extract raw text.</li>
                <li><strong>AI Reasoning:</strong> <code>Ollama (Gemma:2b)</code> analyzes the unstructured text.</li>
                <li><strong>Extraction:</strong> It identifies <em>Conditions</em> (Diabetes), <em>Allergens</em> (Peanut), and <em>Vitals</em> (Glucose: 120mg/dL).</li>
            </ul>
        </div>

        <div class="flow-step">
            <h3>Phase 2: User Profiling (Storage)</h3>
            <p><strong>Process:</strong> The extracted data is merged with user bio-metrics (Age, Height, Weight).</p>
            <p><strong>Output:</strong> A "Medical Profile" stored in <code>SQLite</code> that acts as a filter for all future food interactions.</p>
        </div>

        <div class="flow-step">
            <h3>Phase 3: Smart Tracking (Interaction)</h3>
            <p><strong>Input:</strong> User scans a food item.</p>
            <p><strong>Safety Engine:</strong> The system checks: <em>Does Food X contain ingredients harmful to Condition Y?</em></p>
            <p><strong>Result:</strong> Real-time warning ("High Sugar Risk") or approval.</p>
        </div>

        <div class="flow-step">
            <h3>Phase 4: Generative Recipes (Output)</h3>
            <p><strong>Process:</strong> The AI generates a unique recipe by combining:</p>
            <code>[User Ingredients] + [Medical Constraints] + [Caloric Target]</code>
        </div>

        <h2>2. Technology Stack</h2>
        
        <h3>Backend Infrastructure</h3>
        <table>
            <tr><th>Component</th><th>Technology</th><th>Role</th></tr>
            <tr><td><strong>Language</strong></td><td>Python 3.9+</td><td>Core logic and AI integration.</td></tr>
            <tr><td><strong>API Framework</strong></td><td>FastAPI</td><td>High-performance async endpoints.</td></tr>
            <tr><td><strong>Database</strong></td><td>SQLite</td><td>Local, file-based relational storage.</td></tr>
            <tr><td><strong>Container</strong></td><td>Docker</td><td>Environment isolation.</td></tr>
        </table>

        <h3>Artificial Intelligence</h3>
        <div class="tech-box">
            <p><strong>LLM Engine:</strong> <code>Ollama</code> running <code>gemma:2b</code> (Local, Privacy-focused).</p>
            <p><strong>Vision (OCR):</strong> <code>PaddleOCR</code> (Deep Learning) + <code>Tesseract</code> (Fallback) + <code>pypdf</code>.</p>
        </div>

        <h3>Frontend</h3>
        <p>Built with <strong>Vanilla HTML5/CSS3/JS</strong> for maximum performance and zero build-step complexity. Uses <strong>Glassmorphism</strong> design principles.</p>

        <hr>
        <p style="text-align: center; color: #aaa; font-size: 0.9em;">Generated for Project Documentation</p>
    </div>
</body>
</html>
"""

with open("AI_Nutrition_System_Architecture.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("HTML Documentation Generated.")
