"""
Nutrition Parser Module

Regex-based extraction of nutrition values from OCR text.
Handles various label formats (US, EU, Indian).
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class NutritionSource(Enum):
    """Source of nutrition data."""
    OCR = "ocr"           # Extracted directly from label
    RAG = "rag"           # Retrieved from database
    MANUAL = "manual"     # User entered


@dataclass
class ParsedNutrition:
    """
    Parsed nutrition values from OCR or database.
    All values are per serving unless specified otherwise.
    """
    calories: Optional[float] = None
    carbs_g: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    fiber_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    potassium_mg: Optional[float] = None
    serving_size: Optional[str] = None
    serving_unit: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if we have minimum required fields for logging."""
        return self.calories is not None and (
            self.carbs_g is not None or 
            self.protein_g is not None or 
            self.fat_g is not None
        )
    
    def completeness_score(self) -> float:
        """Calculate how complete the nutrition data is (0-1)."""
        fields = [
            self.calories, self.carbs_g, self.protein_g, 
            self.fat_g, self.sugar_g, self.sodium_mg
        ]
        filled = sum(1 for f in fields if f is not None)
        return filled / len(fields)
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in {
            "calories": self.calories,
            "carbs_g": self.carbs_g,
            "protein_g": self.protein_g,
            "fat_g": self.fat_g,
            "sugar_g": self.sugar_g,
            "sodium_mg": self.sodium_mg,
            "fiber_g": self.fiber_g,
            "saturated_fat_g": self.saturated_fat_g,
            "cholesterol_mg": self.cholesterol_mg,
            "potassium_mg": self.potassium_mg,
            "serving_size": self.serving_size,
            "serving_unit": self.serving_unit,
        }.items() if v is not None}


class NutritionParser:
    """
    Parses nutrition values from OCR text using regex patterns.
    
    Supports multiple label formats:
    - US Nutrition Facts
    - EU Nutrition Information
    - Indian FSSAI labels
    """
    
    # Regex patterns for each nutrient
    # Multiple patterns per nutrient to handle variations
    PATTERNS = {
        "calories": [
            r"calories[:\s]*(\d+)",
            r"energy[:\s]*(\d+)\s*kcal",
            r"(\d+)\s*kcal",
            r"cal[:\s]*(\d+)",
            r"kilocalories[:\s]*(\d+)",
            r"kcal[:\s]*(\d+)",
        ],
        "carbs": [
            r"total\s+carbohydrate[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"carbohydrate[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"carbs[:\s]*(\d+\.?\d*)\s*g",
            r"carb[:\s]*(\d+\.?\d*)",
        ],
        "protein": [
            r"protein[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"proteins[:\s]*(\d+\.?\d*)",
        ],
        "fat": [
            r"total\s+fat[:\s]*(\d+\.?\d*)\s*g",
            r"fat[:\s]*(\d+\.?\d*)\s*g",
            r"fats[:\s]*(\d+\.?\d*)",
        ],
        "saturated_fat": [
            r"saturated\s+fat[:\s]*(\d+\.?\d*)\s*g",
            r"saturated[:\s]*(\d+\.?\d*)\s*g",
            r"sat\.\s*fat[:\s]*(\d+\.?\d*)",
        ],
        "sugar": [
            r"total\s+sugar[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"sugar[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"of\s+which\s+sugars[:\s]*(\d+\.?\d*)",
        ],
        "sodium": [
            r"sodium[:\s]*(\d+\.?\d*)\s*mg",
            r"sodium[:\s]*(\d+\.?\d*)",
            r"na[:\s]*(\d+\.?\d*)\s*mg",
        ],
        "salt": [
            r"salt[:\s]*(\d+\.?\d*)\s*g",  # Convert: salt_g * 400 = sodium_mg
        ],
        "fiber": [
            r"dietary\s+fib[er|re]+[:\s]*(\d+\.?\d*)\s*g",
            r"fibre[:\s]*(\d+\.?\d*)\s*g",
            r"fiber[:\s]*(\d+\.?\d*)\s*g",
        ],
        "cholesterol": [
            r"cholesterol[:\s]*(\d+\.?\d*)\s*mg",
        ],
        "potassium": [
            r"potassium[:\s]*(\d+\.?\d*)\s*mg",
        ],
        "serving_size": [
            r"serving\s+size[:\s]*([^\n\r]+)",
            r"per\s+serving[:\s]*\(?([^\n\r\)]+)\)?",
            r"portion[:\s]*([^\n\r]+)",
        ],
    }
    
    # Keywords that indicate a nutrition facts panel
    PANEL_INDICATORS = [
        "nutrition facts",
        "nutritional information", 
        "nutrition information",
        "per serving",
        "per 100g",
        "per 100ml",
        "amount per serving",
        "calories",
        "protein",
        "carbohydrate",
    ]
    
    def has_nutrition_panel(self, text: str) -> bool:
        """
        Detect if OCR text contains a nutrition facts panel.
        Requires at least 2 indicators to avoid false positives.
        """
        text_lower = text.lower()
        matches = sum(1 for indicator in self.PANEL_INDICATORS 
                     if indicator in text_lower)
        return matches >= 2
    
    def parse(self, text: str) -> ParsedNutrition:
        """
        Parse nutrition values from OCR text.
        
        Args:
            text: Raw OCR text output
            
        Returns:
            ParsedNutrition with extracted values
        """
        text_lower = text.lower()
        nutrition = ParsedNutrition()
        
        # Extract each nutrient
        nutrition.calories = self._extract_value(text_lower, "calories")
        nutrition.carbs_g = self._extract_value(text_lower, "carbs")
        nutrition.protein_g = self._extract_value(text_lower, "protein")
        nutrition.fat_g = self._extract_value(text_lower, "fat")
        nutrition.saturated_fat_g = self._extract_value(text_lower, "saturated_fat")
        nutrition.sugar_g = self._extract_value(text_lower, "sugar")
        nutrition.fiber_g = self._extract_value(text_lower, "fiber")
        nutrition.cholesterol_mg = self._extract_value(text_lower, "cholesterol")
        nutrition.potassium_mg = self._extract_value(text_lower, "potassium")
        
        # Handle sodium (may come from salt)
        sodium = self._extract_value(text_lower, "sodium")
        if sodium is None:
            salt = self._extract_value(text_lower, "salt")
            if salt is not None:
                sodium = salt * 400  # Convert salt (g) to sodium (mg)
        nutrition.sodium_mg = sodium
        
        # Extract serving size
        serving = self._extract_serving(text_lower)
        if serving:
            nutrition.serving_size = serving[0]
            nutrition.serving_unit = serving[1]
        
        return nutrition
    
    def _extract_value(self, text: str, nutrient: str) -> Optional[float]:
        """Extract numeric value for a nutrient using regex patterns."""
        patterns = self.PATTERNS.get(nutrient, [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Basic validation
                    if value >= 0:
                        return value
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_serving(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract serving size and unit."""
        for pattern in self.PATTERNS["serving_size"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                serving_text = match.group(1).strip()
                # Try to parse quantity and unit
                qty_match = re.search(r"(\d+\.?\d*)\s*(g|ml|oz|cup|piece|bar)", 
                                     serving_text, re.IGNORECASE)
                if qty_match:
                    return (qty_match.group(1), qty_match.group(2).lower())
                return (serving_text, "serving")
        
        return None
    
    def extract_food_name(self, text: str) -> Optional[str]:
        """
        Extract food/product name from OCR text.
        Usually appears before nutrition panel or at the top.
        """
        lines = text.strip().split('\n')
        
        # Filter out nutrition-related lines
        nutrition_words = {'nutrition', 'calories', 'fat', 'protein', 'carb', 
                          'serving', 'amount', 'daily', 'value', 'ingredients'}
        
        for line in lines[:5]:  # Check first 5 lines
            line_lower = line.lower().strip()
            # Skip empty or short lines
            if len(line_lower) < 3:
                continue
            # Skip lines with nutrition keywords
            if any(word in line_lower for word in nutrition_words):
                continue
            # This is likely the product name
            return line.strip()
        
        return None


# =============================================================================
# MEDICAL REPORT OCR PARSER
# =============================================================================

# Known medical conditions to detect
KNOWN_CONDITIONS = [
    "diabetes", "diabetic", "type 1 diabetes", "type 2 diabetes",
    "hypertension", "high blood pressure", "hbp", "elevated bp",
    "heart disease", "cardiovascular", "coronary artery disease", "cad",
    "kidney disease", "renal disease", "ckd", "chronic kidney",
    "liver disease", "hepatic", "fatty liver",
    "thyroid", "hypothyroid", "hyperthyroid",
    "obesity", "overweight", "bmi >30",
    "cholesterol", "hyperlipidemia", "high cholesterol",
    "anemia", "low hemoglobin", "iron deficiency",
    "gout", "uric acid", "hyperuricemia",
    "celiac", "gluten intolerance",
    "lactose intolerance", "dairy intolerance",
    "ibs", "irritable bowel",
    "crohn", "ulcerative colitis",
    "gastritis", "gerd", "acid reflux",
]

# Known allergens to detect
KNOWN_ALLERGENS = [
    "peanut", "peanuts", "groundnut",
    "tree nut", "almond", "walnut", "cashew", "pistachio", "hazelnut",
    "milk", "dairy", "lactose",
    "egg", "eggs",
    "wheat", "gluten",
    "soy", "soybean", "soya",
    "fish", "shellfish", "shrimp", "crab", "lobster",
    "sesame",
    "sulfite", "sulphite",
    "mustard",
]


def parse_medical_report(file_path: str) -> dict:
    """
    Parse a medical report file and extract conditions and allergens.
    Strategy: 
    1. Try digital text (PDF only)
    2. Try extracting images from PDF -> OCR
    3. Try full PDF conversion -> OCR
    4. Direct Image OCR
    """
    import os
    print(f"[OCR] Parsing medical report: {file_path}")
    
    raw_text = "--- OCR DEBUG LOG ---\n"
    conditions = []
    allergens = []
    
    # 0. Setup OCR Engines (Global state or per-call)
    # We try in order: PaddleOCR, EasyOCR, Tesseract
    
    ocr_engines = [] # List of initialized engines
    
    # 1. PaddleOCR Attempt
    try:
        from paddleocr import PaddleOCR
        print("[OCR] Initializing PaddleOCR...")
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        ocr_engines.append(("paddle", paddle_ocr))
        raw_text += "[Info] PaddleOCR initialized.\n"
    except Exception as e:
        raw_text += f"[Warning] PaddleOCR failed to init: {e}\n"
    
    # 2. EasyOCR Attempt
    try:
        import easyocr
        print("[OCR] Initializing EasyOCR...")
        easy_ocr_reader = easyocr.Reader(['en'], gpu=False) # GPU False for stability if not sure
        ocr_engines.append(("easyocr", easy_ocr_reader))
        raw_text += "[Info] EasyOCR initialized.\n"
    except Exception as e:
        raw_text += f"[Warning] EasyOCR failed to init: {e}\n"

    # 3. Tesseract Setup
    import pytesseract
    from PIL import Image
    tesseract_available = False
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\hp\AppData\Local\Tesseract-OCR\tesseract.exe',
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            raw_text += f"[Info] Tesseract found at: {path}\n"
            tesseract_available = True
            break
    
    if not tesseract_available:
        raw_text += "[Warning] Tesseract not found in common paths.\n"
    else:
        ocr_engines.append(("tesseract", pytesseract))

    if not ocr_engines:
        raw_text += "[Critical] No OCR engines available!\n"

    def run_multi_ocr(img_pil):
        """Internal helper to run best available OCR on an image."""
        import numpy as np
        
        for engine_type, engine in ocr_engines:
            try:
                if engine_type == "paddle":
                    res = engine.ocr(np.array(img_pil), cls=True)
                    if res and res[0]:
                        return "\n".join([line[1][0] for line in res[0]])
                elif engine_type == "easyocr":
                    res = engine.readtext(np.array(img_pil), detail=0)
                    if res:
                        return "\n".join(res)
                elif engine_type == "tesseract":
                    return engine.image_to_string(img_pil)
            except Exception as e:
                print(f"[OCR] Engine {engine_type} failed during processing: {e}")
        return ""

    try:
        if file_path.lower().endswith('.pdf'):
            raw_text += "[Info] Detected PDF file format.\n"
            import pypdf
            reader = pypdf.PdfReader(file_path)
            
            # 1. Digital Text
            digital_text = ""
            for i, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt: digital_text += txt + "\n"
            
            if len(digital_text.strip()) > 50:
                raw_text += f"[Success] Extracted {len(digital_text)} chars from digital layer.\n"
                raw_text += digital_text
                extracted_success = True
            else:
                raw_text += "[Info] No digital text found, scanning for images...\n"
                
                # 2. Embedded Images
                img_extracted_text = ""
                for i, page in enumerate(reader.pages):
                    for img_obj in page.images:
                        try:
                            import io
                            img = Image.open(io.BytesIO(img_obj.data))
                            txt = run_multi_ocr(img)
                            if txt: img_extracted_text += txt + "\n"
                        except Exception as e:
                            raw_text += f"[Error] Image on page {i+1} skip: {e}\n"
                
                if img_extracted_text.strip():
                    raw_text += "[Success] Extracted text from embedded PDF images.\n"
                    raw_text += img_extracted_text
                    extracted_success = True
                else:
                    # 3. Full Page conversion (Requires Poppler)
                    raw_text += "[Info] No embedded images found, trying full page conversion...\n"
                    try:
                        from pdf2image import convert_from_path
                        pages = convert_from_path(file_path)
                        for page_img in pages:
                            txt = run_multi_ocr(page_img)
                            if txt: raw_text += txt + "\n"
                        extracted_success = True
                    except Exception as e:
                        raw_text += f"[Error] pdf2image failed (likely missing poppler): {e}\n"

        else: # Image file
            raw_text += "[Info] Detected Image file format.\n"
            img = Image.open(file_path)
            txt = run_multi_ocr(img)
            
            if txt:
                raw_text += txt
                extracted_success = True
            else:
                raw_text += "[Error] Failed to extract any text from image.\n"

    except Exception as e:
        raw_text += f"[Critical] Error in OCR pipeline: {e}\n"

    print(f"[OCR] Extracted {len(raw_text)} chars. Preview: {raw_text[:500]}...")

    # Pattern Matching (Improved with word boundaries and bi-directional negation detection)
    text_lower = raw_text.lower()
    NEGATIONS = ["no", "none", "negative", "not", "without", "denies", "denied", "absent"]
    
    for condition in KNOWN_CONDITIONS:
        # Use word boundaries for precise matching
        pattern = r'\b' + re.escape(condition) + r'\b'
        matches = list(re.finditer(pattern, text_lower))
        
        for match in matches:
            start = match.start()
            end = match.end()
            
            # 1. Look back 40 characters for negations
            prefix = text_lower[max(0, start-40):start]
            # 2. Look ahead 20 characters for negations like ": no" or "- negative"
            suffix = text_lower[end:min(len(text_lower), end+20)]
            
            is_negated = False
            for neg in NEGATIONS:
                neg_pattern = r'\b' + re.escape(neg) + r'\b'
                if re.search(neg_pattern, prefix) or re.search(neg_pattern, suffix):
                    is_negated = True
                    break
            
            if not is_negated:
                # Normalize
                normalized = condition.replace("type 1 diabetes", "Type 1 Diabetes").replace("type 2 diabetes", "Type 2 Diabetes")
                normalized = normalized.replace("diabetic", "Diabetes").replace("diabetes", "Diabetes")
                normalized = normalized.replace("hypertension", "Hypertension").replace("high blood pressure", "Hypertension")
                
                cond_title = normalized.title()
                if cond_title not in conditions:
                    conditions.append(cond_title)
                break

    for allergen in KNOWN_ALLERGENS:
        pattern = r'\b' + re.escape(allergen) + r'\b'
        if re.search(pattern, text_lower):
            normalized = allergen.title()
            if "peanut" in allergen: normalized = "Peanuts"
            elif "nut" in allergen: normalized = "Tree Nuts"
            elif "milk" in allergen or "dairy" in allergen: normalized = "Dairy"
            if normalized not in allergens: allergens.append(normalized)

    vitals = {}
    biometrics = {}
    
    # 1. Glucose & HbA1c
    match_gl = re.search(r'(?:glucose|fasting glucose|sugar)[^0-9]*(\d{1,3}\.?\d?)', text_lower)
    if match_gl: vitals["glucose_level"] = float(match_gl.group(1))
    
    match_hba1c = re.search(r'hba1c[^0-9]*(\d{1,2}\.?\d?)', text_lower)
    if match_hba1c: vitals["hba1c"] = float(match_hba1c.group(1))
    
    # 2. Blood Pressure
    match_bp = re.search(r'bp[^0-9]*(\d{2,3})[/\s]*(\d{2,3})', text_lower)
    if match_bp:
        vitals["systolic_bp"] = float(match_bp.group(1))
        vitals["diastolic_bp"] = float(match_bp.group(2))
    
    # 3. Lipid Profile (Cholesterol, Triglycerides, HDL, LDL)
    match_ch = re.search(r'cholesterol[^0-9]*(\d{2,4}\.?\d?)', text_lower)
    if match_ch: vitals["cholesterol"] = float(match_ch.group(1))
    
    match_tg = re.search(r'triglycerides[^0-9]*(\d{1,4}\.?\d?)', text_lower)
    if match_tg: vitals["triglycerides"] = float(match_tg.group(1))
    
    match_hdl = re.search(r'hdl[^0-9]*(\d{1,3}\.?\d?)', text_lower)
    if match_hdl: vitals["hdl"] = float(match_hdl.group(1))
    
    match_ldl = re.search(r'ldl[^0-9]*(\d{1,3}\.?\d?)', text_lower)
    if match_ldl: vitals["ldl"] = float(match_ldl.group(1))

    # 4. Biometrics
    match_age = re.search(r'\bage\b[^0-9]*(\d{1,2})', text_lower)
    if match_age: 
        biometrics["age"] = int(match_age.group(1))
    else:
        # Try finding Date of Birth (e.g., 01.01.1973)
        match_dob = re.search(r'(?:dob|birth)[^0-9]*(\d{2}[\./-]\d{2}[\./-]\d{4})', text_lower)
        if match_dob:
            try:
                from datetime import datetime
                dob_str = match_dob.group(1).replace('.', '-').replace('/', '-')
                dob = datetime.strptime(dob_str, "%d-%m-%Y")
                age = datetime.now().year - dob.year
                biometrics["age"] = age
            except:
                pass

    match_weight = re.search(r'(?:weight|wt|body weight)[^0-9]*(\d{2,3}\.?\d*)', text_lower)
    if match_weight: biometrics["weight_kg"] = float(match_weight.group(1))
    
    match_height = re.search(r'(?:height|ht)[^0-9]*(\d{2,3}\.?\d*)', text_lower)
    if match_height: biometrics["height_cm"] = float(match_height.group(1))
    
    if "female" in text_lower: biometrics["gender"] = "female"
    elif "male" in text_lower: biometrics["gender"] = "male"

    # 5. Medications
    medications = []
    common_meds = [
        "metformin", "insulin", "atorvastatin", "amlodipine", "losartan",
        "levothyroxine", "lisinopril", "gabapentin", "metoprolol", "albuterol",
        "paracetamol", "ibuprofen", "aspirin", "omeprazole", "glimepiride",
        "sitagliptin", "telmisartan", "rosuvastatin", "vildagliptin"
    ]
    
    for med in common_meds:
        pattern = r'\b' + re.escape(med) + r'\b'
        if re.search(pattern, text_lower):
            med_title = med.title()
            if med_title not in medications:
                medications.append(med_title)

    return {
        "raw_text": raw_text,
        "conditions": conditions,
        "allergens": allergens,
        "vitals": vitals,
        "biometrics": biometrics,
        "medications": medications
    }
