
import sys

print("--- DEBUGGING START ---")

print("1. Checking pypdf...")
try:
    import pypdf
    print(f"   [SUCCESS] pypdf version: {pypdf.__version__}")
except ImportError as e:
    print(f"   [FAILURE] pypdf import failed: {e}")
except Exception as e:
    print(f"   [FAILURE] pypdf error: {e}")

print("\n2. Checking PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    print("   [INFO] Importing successful. Initializing engine (may take time)...")
    # minimal init
    ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    print("   [SUCCESS] PaddleOCR initialized!")
except ImportError as e:
    print(f"   [FAILURE] PaddleOCR import failed: {e}")
except Exception as e:
    print(f"   [FAILURE] PaddleOCR error: {e}")

print("\n3. Checking pdf2image...")
try:
    from pdf2image import convert_from_path
    print("   [SUCCESS] pdf2image imported (Poppler check happens at runtime)")
except ImportError:
    print("   [FAILURE] pdf2image not installed")

print("\n4. Checking EasyOCR...")
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    print("   [SUCCESS] EasyOCR initialized!")
except ImportError:
    print("   [FAILURE] EasyOCR not installed")
except Exception as e:
    print(f"   [FAILURE] EasyOCR error: {e}")

print("\n5. Testing new multi-engine parser (Dry Run)...")
try:
    from ocr.parser import parse_medical_report
    # We won't pass a real file yet, just check if it imports and initializes
    print("   [SUCCESS] Parser imported successfully.")
except Exception as e:
    print(f"   [FAILURE] Parser error: {e}")

print("--- DEBUGGING END ---")
