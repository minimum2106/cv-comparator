import re
import unicodedata
from typing import Optional

class TextProcessor:
    """Utility class for handling text encoding and cleaning issues"""
    
    # Common encoding fixes for French characters
    ENCODING_FIXES = {
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ãª': 'ê',
        'Ã ': 'à',
        'Ã¢': 'â',
        'Ã´': 'ô',
        'Ã¯': 'ï',
        'Ã®': 'î',
        'Ã§': 'ç',
        'Ã¹': 'ù',
        'Ã»': 'û',
        'Ã¼': 'ü',
        'Ã«': 'ë',
        'Ã±': 'ñ',
        'Ã': 'À',
        'Ã‰': 'É',
        'Ã': 'È',
        'ÃŠ': 'Ê',
        'Ã‡': 'Ç',
        'Ã¡': 'á',
        'Ã­': 'í',
        'Ã³': 'ó',
        'Ãº': 'ú',
        'Ã½': 'ý',
        'Ã': 'Á',
        'Ã': 'Í',
        'Ã"': 'Ó',
        'Ãš': 'Ú',
        'Ã': 'Ý',
        # Common multi-character encodings
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '–',
        'â€"': '—',
        'â€¦': '…',
        'Â ': ' ',
        'Â': '',
    }

    @classmethod
    def fix_encoding(cls, text: str) -> str:
        """Fix common encoding issues in text"""
        if not text:
            return text
        
        # Apply encoding fixes
        for wrong, correct in cls.ENCODING_FIXES.items():
            text = text.replace(wrong, correct)
        
        return text

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return text
        
        # Fix encoding first
        text = cls.fix_encoding(text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '').replace('\ufffd', '')
        
        return text

    @classmethod
    def detect_and_fix_encoding(cls, text: str) -> str:
        """Detect encoding issues and fix them"""
        if not text:
            return text
        
        # Check if text contains encoding artifacts
        encoding_indicators = ['Ã', 'â€', 'Â']
        has_encoding_issues = any(indicator in text for indicator in encoding_indicators)
        
        if has_encoding_issues:
            print("🔧 Detected encoding issues, fixing...")
            text = cls.fix_encoding(text)
            print("✅ Encoding fixed")
        
        return cls.clean_text(text)

    @classmethod
    def safe_json_string(cls, text: str) -> str:
        """Prepare text for safe JSON encoding"""
        if not text:
            return text
        
        # Fix encoding and clean
        text = cls.detect_and_fix_encoding(text)
        
        # Escape quotes and backslashes for JSON
        text = text.replace('\\', '\\\\').replace('"', '\\"')
        
        return text