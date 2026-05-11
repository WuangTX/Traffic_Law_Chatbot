"""
Module: metadata_extractor.py
Purpose: Extract metadata from legal text (penalties, points, vehicle type, references)
Uses regex patterns to identify key legal concepts for RAG filtering.
"""

import re
from typing import Dict, List, Optional, Tuple, Any


class MetadataExtractor:
    """Extract structured metadata from Vietnamese legal text."""
    
    # Vehicle type mapping from article titles (Nghị định 168)
    VEHICLE_TYPE_PATTERNS = {
        r'(?:xe\s+)?ô\s*tô|người\s+điều\s+khiển\s+xe\s+ô\s*tô|xử\s+phạt.*?ô\s*tô': 'ô tô',
        r'xe\s+máy|xe\s+gắn\s+máy|người\s+điều\s+khiển\s+xe\s+máy': 'xe máy',
        r'xe\s+mô\s+tô|mô\s+tô.*?bánh': 'mô tô',
        r'xe\s+đạp\s+motor|xe\s+đạp': 'xe đạp',
        r'xe\s+thô\s+sơ': 'xe thô sơ',
    }
    
    # Penalty patterns (tìm cụm "phạt X đến Y đồng")
    # Handles Vietnamese format: 400.000 (meaning 400,000), or 1 triệu, or 100k, etc.
    PENALTY_PATTERN = r'phạt\s+(?:tiền\s+)?(?:từ\s+)?(\d+(?:\.\d{3})?)\s*(?:triệu\s+)?(?:đồng|nghìn|k)(?:\s+đến\s+|[\s-]+)(\d+(?:\.\d{3})?)\s*(?:triệu\s+)?(?:đồng|nghìn|k)?'
    
    # Points deduction patterns (tìm cụm "trừ X điểm")
    POINTS_PATTERN = r'trừ\s+(?:đi\s+)?(\d+)\s*(?:điểm|points?)'
    
    # License suspension patterns (tạm dừng bằng)
    SUSPENSION_PATTERN = r'(?:tạm\s+)?(?:dừng|ấu|hủy|thu|cấp|cấp\s+lại)\s+(?:giấy|bằng)\s+(?:lái\s+xe|cấp)\s+(?:từ\s+)?(\d+)\s*(?:tháng|năm|ngày)'
    
    # Cross-reference patterns (Khoản X Điều Y, Điểm Z, v.v.)
    REFERENCE_PATTERNS = [
        r'Khoản\s+(\d+)\s+Điều\s+(\d+)',           # Khoản 1 Điều 6
        r'Điểm\s+([a-zđ])\s+(?:Khoản|khoản)?\s*(\d*)\s+Điều\s+(\d+)',  # Điểm a Khoản 1 Điều 6
        r'Điều\s+(\d+)',                             # Điều 6 (simple)
    ]
    
    # Date patterns (tìm ngày có hiệu lực)
    EFFECTIVE_DATE_PATTERN = r'(?:có\s+hiệu\s+lực|từ\s+ngày|kể\s+từ\s+ngày|áp\s+dụng)\s+(?:từ\s+)?(?:ngày\s+)?(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
    
    # Article title keywords for vehicle type detection
    ARTICLE_VEHICLE_KEYWORDS = {
        ('ô tô', 'xe ô tô'): 'ô tô',
        ('xe máy', 'xe gắn máy'): 'xe máy',
        ('mô tô', 'xe mô tô'): 'mô tô',
        ('xe đạp',): 'xe đạp',
        ('xe thô sơ',): 'xe thô sơ',
    }
    
    @classmethod
    def extract_metadata(cls, 
                        text: str, 
                        article_title: str = '',
                        article_number: int = 0,
                        law_source: str = '') -> Dict[str, Any]:
        """
        Extract all metadata from legal text.
        
        Args:
            text: Legal text to extract from
            article_title: Title of the article (helps identify vehicle type)
            article_number: Article number (Điều)
            law_source: Source law/regulation (e.g., "Nghị định 168/2024/NĐ-CP")
            
        Returns:
            Dictionary with extracted metadata
        """
        return {
            'vehicle_type': cls.extract_vehicle_type(article_title, text),
            'penalty_range': cls.extract_penalty_range(text),
            'points_deducted': cls.extract_points_deducted(text),
            'license_suspension': cls.extract_license_suspension(text),
            'references': cls.extract_references(text),
            'effective_date': cls.extract_effective_date(text),
            'article_number': article_number,
            'law_source': law_source,
        }
    
    @classmethod
    def extract_vehicle_type(cls, article_title: str, text: str = '') -> Optional[str]:
        """
        Detect vehicle type from article title or text.
        
        Examples:
        - "Xử phạt người điều khiển xe ô tô vi phạm..." → "ô tô"
        - "Xử phạt người điều khiển xe máy..." → "xe máy"
        - "Xe mô tô..." → "mô tô"
        """
        # Try article title first (more reliable)
        combined = f"{article_title} {text}".lower()
        
        for keywords, vehicle in cls.ARTICLE_VEHICLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in combined:
                    return vehicle
        
        # Fallback to regex patterns
        for pattern, vehicle in cls.VEHICLE_TYPE_PATTERNS.items():
            if re.search(pattern, combined, re.IGNORECASE):
                return vehicle
        
        return None
    
    @classmethod
    def extract_penalty_range(cls, text: str) -> Optional[List[int]]:
        """
        Extract penalty range (minimum and maximum).
        Returns: [min_penalty_vnd, max_penalty_vnd] or None
        
        Examples:
        - "phạt tiền từ 400.000 đồng đến 600.000 đồng" → [400000, 600000]
        - "phạt 1-2 triệu" → [1000000, 2000000]
        - "phạt từ 100k-200k" → [100000, 200000]
        """
        text_lower = text.lower()
        matches = re.finditer(cls.PENALTY_PATTERN, text_lower)
        
        penalties = []
        for match in matches:
            # Extract and normalize Vietnamese number format (dots are thousand separators)
            min_str = match.group(1).replace('.', '')  # "400.000" → "400000"
            max_str = match.group(2).replace('.', '')  # "600.000" → "600000"
            
            min_val = float(min_str)
            max_val = float(max_str)
            
            # Normalize to VND (check if in triệu or nghìn)
            if 'triệu' in match.group(0):
                min_val *= 1_000_000
                max_val *= 1_000_000
            elif 'k' in match.group(0) or 'nghìn' in match.group(0):
                min_val *= 1_000
                max_val *= 1_000
            
            penalties.append([int(min_val), int(max_val)])
        
        return penalties[0] if penalties else None
    
    @classmethod
    def extract_points_deducted(cls, text: str) -> Optional[List[int]]:
        """
        Extract point deductions (trừ điểm).
        Returns: List of point values or None
        
        Examples:
        - "trừ 02 điểm" → [2]
        - "trừ 04 hoặc 06 điểm" → [4, 6]
        - "trừ 10 điểm" → [10]
        """
        text_lower = text.lower()
        matches = re.finditer(cls.POINTS_PATTERN, text_lower)
        
        points = []
        for match in matches:
            points.append(int(match.group(1)))
        
        return list(set(points)) if points else None
    
    @classmethod
    def extract_license_suspension(cls, text: str) -> Optional[Tuple[int, str]]:
        """
        Extract license suspension duration.
        Returns: (duration, unit) e.g., (3, 'tháng') or None
        
        Examples:
        - "tạm dừng bằng lái từ 1 tháng" → (1, 'tháng')
        - "dừng giấy phép từ 6 tháng" → (6, 'tháng')
        """
        text_lower = text.lower()
        match = re.search(cls.SUSPENSION_PATTERN, text_lower)
        
        if match:
            duration = int(match.group(1))
            # Detect unit
            if 'năm' in match.group(0):
                unit = 'năm'
            elif 'ngày' in match.group(0):
                unit = 'ngày'
            else:
                unit = 'tháng'
            return (duration, unit)
        
        return None
    
    @classmethod
    def extract_references(cls, text: str) -> List[str]:
        """
        Extract cross-references to other articles/clauses.
        Returns: List of references like ["Điều 6", "Khoản 1 Điều 7", etc.]
        
        Examples:
        - "Khoản 16 Điều 6 quy định..." → ["Khoản 16 Điều 6"]
        - "Xem Điều 5 và Điều 7" → ["Điều 5", "Điều 7"]
        """
        references = []
        
        # Pattern 1: Khoản X Điều Y
        for match in re.finditer(r'Khoản\s+(\d+)\s+(?:Điều|điều)\s+(\d+)', text):
            ref = f"Khoản {match.group(1)} Điều {match.group(2)}"
            references.append(ref)
        
        # Pattern 2: Điểm a Khoản X Điều Y
        for match in re.finditer(r'Điểm\s+([a-zđ])\s+(?:Khoản|khoản)\s+(\d+)\s+(?:Điều|điều)\s+(\d+)', text):
            ref = f"Điểm {match.group(1)} Khoản {match.group(2)} Điều {match.group(3)}"
            references.append(ref)
        
        # Pattern 3: Standalone Điều references (be careful with false positives)
        for match in re.finditer(r'(?:Điều|điều)\s+(\d+)', text):
            ref = f"Điều {match.group(1)}"
            if ref not in references:
                references.append(ref)
        
        return list(set(references))  # Remove duplicates
    
    @classmethod
    def extract_effective_date(cls, text: str) -> Optional[str]:
        """
        Extract effective date of regulation.
        Returns: Date string "YYYY-MM-DD" or None
        
        Examples:
        - "có hiệu lực từ ngày 1/1/2025" → "2025-01-01"
        - "áp dụng từ 01/01/2025" → "2025-01-01"
        """
        match = re.search(cls.EFFECTIVE_DATE_PATTERN, text)
        
        if match:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            return f"{year:04d}-{month:02d}-{day:02d}"
        
        # Fallback: hardcoded dates for Luật 35/2024 and Nghị định 168/2024
        if '35/2024' in text or '35-2024' in text or 'Luật' in text and '2025' in text:
            return "2025-01-01"
        if '168/2024' in text or '168-2024' in text or 'Nghị định' in text and '2025' in text:
            return "2025-01-01"
        
        return None


def test_metadata_extraction():
    """Test metadata extraction with sample texts."""
    
    test_cases = [
        {
            'text': 'Xử phạt người điều khiển xe ô tô. Phạt tiền từ 400.000 đồng đến 600.000 đồng. Trừ 02 điểm.',
            'article_title': 'Xử phạt người điều khiển xe ô tô vi phạm quy tắc giao thông',
            'law_source': 'Nghị định 168/2024/NĐ-CP',
        },
        {
            'text': 'Xử phạt người điều khiển xe máy. Phạt từ 200k đến 400k. Trừ 04 điểm. Xem Khoản 1 Điều 6.',
            'article_title': 'Xử phạt người điều khiển xe máy vi phạm quy tắc giao thông',
            'law_source': 'Nghị định 168/2024/NĐ-CP',
        },
    ]
    
    print("=" * 80)
    print("METADATA EXTRACTION TEST")
    print("=" * 80)
    
    for idx, case in enumerate(test_cases, 1):
        print(f"\nTest case {idx}:")
        print(f"Article: {case['article_title']}")
        print(f"Text: {case['text'][:80]}...")
        
        metadata = MetadataExtractor.extract_metadata(
            text=case['text'],
            article_title=case['article_title'],
            article_number=6 if idx == 1 else 7,
            law_source=case['law_source']
        )
        
        print(f"Vehicle type: {metadata['vehicle_type']}")
        print(f"Penalty range: {metadata['penalty_range']} VND")
        print(f"Points deducted: {metadata['points_deducted']}")
        print(f"References: {metadata['references']}")
        print(f"Effective date: {metadata['effective_date']}")


if __name__ == '__main__':
    test_metadata_extraction()
