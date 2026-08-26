"""
Report Parser — Extract ECG parameters from PDF/Image/Text reports.
Supports:
- PDF files (using pypdf)
- Image files (using pytesseract OCR)
- Plain text
"""

import os
import re
from typing import Optional, Dict, Any, Union
import tempfile

from clinical_report import ECGReportData, parse_report_text


# ============================================================================
# PDF PARSING
# ============================================================================

def parse_pdf_report(file_path: str) -> Optional[ECGReportData]:
    """
    Parse ECG report from PDF file path.
    Extracts text using pypdf and then parses measurements.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        ECGReportData object if successful, None otherwise
    """
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            return None
        
        return parse_report_text(text)
        
    except ImportError:
        return None
    except Exception as e:
        return None


def parse_pdf_file_object(file_like) -> Optional[ECGReportData]:
    """
    Parse ECG report from a file-like object (e.g., Streamlit UploadedFile).
    
    Args:
        file_like: File-like object with a .read() method
        
    Returns:
        ECGReportData object if successful, None otherwise
    """
    try:
        from pypdf import PdfReader
        
        # Reset file pointer if needed
        if hasattr(file_like, 'seek'):
            file_like.seek(0)
        
        reader = PdfReader(file_like)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            return None
        
        return parse_report_text(text)
        
    except ImportError:
        return None
    except Exception as e:
        return None


# ============================================================================
# IMAGE / OCR PARSING
# ============================================================================

def parse_image_report(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse ECG report from image file using OCR (Tesseract).
    
    Requires: pip install pytesseract pillow
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary with extracted ECG parameters, or None if failed
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Open and preprocess image
        img = Image.open(file_path)
        
        # Convert to grayscale for better OCR
        if img.mode != 'L':
            img = img.convert('L')
        
        # Extract text using OCR
        text = pytesseract.image_to_string(img)
        
        if not text.strip():
            return None
        
        # Parse the extracted text
        ecg_data = parse_report_text(text)
        if ecg_data:
            return ecg_data.to_dict()
        return None
        
    except ImportError as e:
        return {"error": f"OCR library not installed. Install: pip install pytesseract pillow. Error: {e}"}
    except Exception as e:
        return {"error": f"Image OCR failed: {e}"}


def parse_image_file_object(file_like) -> Optional[Dict[str, Any]]:
    """
    Parse ECG report from an image file-like object using OCR.
    
    Args:
        file_like: File-like object (e.g., Streamlit UploadedFile)
        
    Returns:
        Dictionary with extracted ECG parameters, or error dict
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Reset file pointer if needed
        if hasattr(file_like, 'seek'):
            file_like.seek(0)
        
        # Open image from file-like object
        img = Image.open(file_like)
        
        # Convert to grayscale for better OCR
        if img.mode != 'L':
            img = img.convert('L')
        
        # Extract text using OCR
        text = pytesseract.image_to_string(img)
        
        if not text.strip():
            return None
        
        # Parse the extracted text
        ecg_data = parse_report_text(text)
        if ecg_data:
            return ecg_data.to_dict()
        return None
        
    except ImportError as e:
        return {"error": f"OCR library not installed. Install: pip install pytesseract pillow. Error: {e}"}
    except Exception as e:
        return {"error": f"Image OCR failed: {e}"}


def parse_image_with_preprocessing(file_path: str, preprocess: bool = True) -> Optional[Dict[str, Any]]:
    """
    Parse ECG report from image with optional preprocessing for better OCR results.
    
    Args:
        file_path: Path to the image file
        preprocess: Whether to apply image preprocessing (thresholding, denoising)
        
    Returns:
        Dictionary with extracted ECG parameters, or None if failed
    """
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter
        
        # Open image
        img = Image.open(file_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        if preprocess:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            
            # Apply sharpening filter
            img = img.filter(ImageFilter.SHARPEN)
            
            # Apply thresholding (binarization)
            threshold = 128
            img = img.point(lambda p: p > threshold and 255)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(img, config='--psm 6')
        
        if not text.strip():
            return None
        
        ecg_data = parse_report_text(text)
        if ecg_data:
            return ecg_data.to_dict()
        return None
        
    except ImportError as e:
        return {"error": f"OCR library not installed. Install: pip install pytesseract pillow. Error: {e}"}
    except Exception as e:
        return {"error": f"Image OCR failed: {e}"}


# ============================================================================
# TEXT PARSING
# ============================================================================

def parse_text_report(text: str) -> ECGReportData:
    """
    Parse ECG report from plain text.
    
    Args:
        text: Raw report text
        
    Returns:
        ECGReportData object with extracted measurements
    """
    return parse_report_text(text)


def extract_ecg_parameters_from_text(text: str) -> Dict[str, Any]:
    """
    Extract ECG parameters from text using regex patterns.
    Returns a dictionary of found parameters for manual pre-fill.
    
    Args:
        text: Raw report text
        
    Returns:
        Dictionary with extracted parameters
    """
    params = {}
    
    # Heart Rate
    hr_match = re.search(r'(?:HR|Heart Rate|Heart rate)[:\s]+(\d+)', text, re.IGNORECASE)
    if hr_match:
        params["heart_rate"] = float(hr_match.group(1))
    
    # PR Interval
    pr_match = re.search(r'(?:PR|PR interval|P-R)[:\s]+(\d+)', text, re.IGNORECASE)
    if pr_match:
        params["pr_interval"] = float(pr_match.group(1))
    
    # QRS Duration
    qrs_match = re.search(r'(?:QRS|QRS duration)[:\s]+(\d+)', text, re.IGNORECASE)
    if qrs_match:
        params["qrs_duration"] = float(qrs_match.group(1))
    
    # QT Interval
    qt_match = re.search(r'(?:QT|QT interval)[:\s]+(\d+)', text, re.IGNORECASE)
    if qt_match:
        params["qt_interval"] = float(qt_match.group(1))
    
    # QTc Interval
    qtc_match = re.search(r'(?:QTc|QTc interval|QTcB|QTcF)[:\s]+(\d+)', text, re.IGNORECASE)
    if qtc_match:
        params["qtc_interval"] = float(qtc_match.group(1))
    
    # Rhythm
    rhythm_patterns = [
        (r'(sinus rhythm|sinus)', 'Sinus'),
        (r'(atrial fibrillation|a-fib|afib)', 'Atrial Fibrillation'),
        (r'(atrial flutter|a-flutter|aflutter)', 'Atrial Flutter'),
        (r'(ventricular tachycardia|vtach)', 'Ventricular Tachycardia'),
        (r'(bradycardia|brady)', 'Bradycardia'),
        (r'(tachycardia|tachy)', 'Tachycardia'),
    ]
    for pattern, value in rhythm_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            params["rhythm"] = value
            break
    
    # Axis
    axis_patterns = [
        (r'(normal axis)', 'Normal'),
        (r'(left axis deviation|left axis|lax)', 'Left Axis Deviation'),
        (r'(right axis deviation|right axis|rax)', 'Right Axis Deviation'),
        (r'(extreme axis|extreme deviation)', 'Extreme Axis Deviation'),
    ]
    for pattern, value in axis_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            params["axis"] = value
            break
    
    # ST Segment
    st_patterns = [
        (r'(st elevation|ste)', 'Elevation'),
        (r'(st depression|std)', 'Depression'),
        (r'(normal st|st normal)', 'Normal'),
    ]
    for pattern, value in st_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            params["st_segment"] = value
            break
    
    # T Wave
    t_patterns = [
        (r'(t wave inversion|t inversion|inverted t|negative t)', 'Inversion'),
        (r'(flat t|t wave flat)', 'Flat'),
        (r'(biphasic t|t wave biphasic|biphasic)', 'Biphasic'),
        (r'(normal t|t wave normal)', 'Normal'),
    ]
    for pattern, value in t_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            params["t_wave"] = value
            break
    
    # Bundle Branch
    if re.search(r'lbbb|left bundle branch block', text, re.IGNORECASE):
        params["bundle_branch"] = 'LBBB'
    elif re.search(r'rbbb|right bundle branch block', text, re.IGNORECASE):
        params["bundle_branch"] = 'RBBB'
    
    # Machine Interpretation
    interp_match = re.search(
        r'(?:Interpretation|Conclusion|Impressions|Findings|Summary)[:\s]+([^\n]+(?:\n[^\n]+)*)', 
        text, 
        re.IGNORECASE
    )
    if interp_match:
        params["machine_interpretation"] = interp_match.group(1).strip()
    
    return params


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_report_format(file_path: str) -> str:
    """
    Detect the format of an ECG report file based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        'pdf', 'image', 'text', or 'unknown'
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    pdf_exts = ['.pdf']
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif']
    text_exts = ['.txt', '.csv', '.text']
    
    if ext in pdf_exts:
        return 'pdf'
    elif ext in image_exts:
        return 'image'
    elif ext in text_exts:
        return 'text'
    else:
        return 'unknown'


def is_ocr_available() -> bool:
    """
    Check if OCR libraries (pytesseract, pillow) are available.
    
    Returns:
        True if OCR is available, False otherwise
    """
    try:
        import pytesseract
        from PIL import Image
        return True
    except ImportError:
        return False


def get_ocr_status() -> Dict[str, bool]:
    """
    Get detailed status of OCR dependencies.
    
    Returns:
        Dictionary with status of each dependency
    """
    status = {
        'pytesseract': False,
        'pillow': False,
        'tesseract_engine': False,
        'ocr_available': False
    }
    
    try:
        import pytesseract
        status['pytesseract'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        status['pillow'] = True
    except ImportError:
        pass
    
    # Check if tesseract engine is accessible
    if status['pytesseract']:
        try:
            import pytesseract
            # Try to get tesseract version
            version = pytesseract.get_tesseract_version()
            status['tesseract_engine'] = True
        except:
            pass
    
    status['ocr_available'] = all([
        status['pytesseract'],
        status['pillow'],
        status['tesseract_engine']
    ])
    
    return status