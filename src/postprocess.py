from __future__ import annotations

import re
from typing import Optional, Dict, List
from collections import Counter


def clean_text(raw_text: str) -> str:
    """
    Post-process OCR output for clean, readable text.
    
    Steps:
    1. Normalize whitespace (collapse multiple spaces/newlines)
    2. Fix common OCR artifacts
    3. Preserve paragraph breaks
    4. Fix punctuation spacing
    5. Remove noise characters
    """
    if not raw_text:
        return ""
    
    text = raw_text
    
    # 1. Normalize line breaks (collapse 3+ newlines to 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 2. Remove trailing/leading whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # 3. Fix broken words across lines (e.g., "доку-\nмент" → "документ")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # 4. Collapse single newlines within sentences (but keep paragraph breaks)
    # If line ends without punctuation and next starts lowercase → join
    def join_lines(match):
        line1 = match.group(1)
        line2 = match.group(2)
        # Keep break if line1 ends with sentence terminator or line2 starts uppercase
        if line1[-1] in '.!?:;' or (line2 and line2[0].isupper()):
            return line1 + '\n' + line2
        # Keep break if line is very short (likely header/title)
        if len(line1) < 40:
            return line1 + '\n' + line2
        # Otherwise join with space
        return line1 + ' ' + line2
    
    text = re.sub(r'([^\n])\n([^\n])', join_lines, text)
    
    # 5. Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,;:!?])([^\s\d])', r'\1 \2', text)  # Add space after punctuation
    
    # 6. Collapse multiple spaces to one
    text = re.sub(r' {2,}', ' ', text)
    
    # 7. Remove common OCR noise patterns
    # Single isolated letters/numbers on own line (likely artifacts)
    text = re.sub(r'\n[a-zA-Zа-яА-ЯёЁ0-9]\n', '\n', text)
    
    # 8. Fix quote spacing
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    
    # 9. Final cleanup
    text = text.strip()
    
    return text


def extract_tables_as_text(text: str) -> str:
    """
    Detect table-like structures and format them cleanly.
    
    Heuristic: Lines with multiple whitespace-separated columns
    """
    lines = text.split('\n')
    result = []
    in_table = False
    table_buffer = []
    
    for line in lines:
        # Detect table row: 3+ segments separated by 2+ spaces
        segments = [s for s in re.split(r'\s{2,}', line.strip()) if s]
        
        if len(segments) >= 3:
            # Likely a table row
            if not in_table:
                in_table = True
                if result:
                    result.append('')  # Add spacing before table
            table_buffer.append(' | '.join(segments))
        else:
            # Not a table row
            if in_table and table_buffer:
                # End of table, add separator
                result.extend(table_buffer)
                result.append('')  # Add spacing after table
                table_buffer = []
                in_table = False
            result.append(line)
    
    # Flush remaining table
    if table_buffer:
        result.extend(table_buffer)
    
    return '\n'.join(result)


def deduplicate_lines(text: str, threshold: int = 3, min_line_length: int = 10) -> str:
    """
    Removes lines that appear more than threshold times (likely noise/watermarks).
    
    Args:
        text: Input text
        threshold: Maximum occurrences allowed (lines appearing > threshold are removed)
        min_line_length: Minimum line length to consider for deduplication
    
    Returns:
        Text with duplicate lines removed
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    
    # Normalize lines for comparison (strip whitespace, lowercase)
    def normalize(line: str) -> str:
        return ' '.join(line.strip().lower().split())
    
    # Count occurrences of normalized lines
    line_counts = Counter()
    original_lines = {}
    
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= min_line_length:
            norm = normalize(line)
            line_counts[norm] += 1
            if norm not in original_lines:
                original_lines[norm] = stripped
    
    # Build result, keeping lines that appear <= threshold times
    result = []
    seen = Counter()
    
    for line in lines:
        stripped = line.strip()
        if len(stripped) < min_line_length:
            result.append(line)
            continue
        
        norm = normalize(line)
        total_count = line_counts[norm]
        
        if total_count <= threshold:
            # Keep all occurrences if below threshold
            result.append(line)
        else:
            # For high-frequency lines, keep only first occurrence
            if seen[norm] == 0:
                result.append(line)
            seen[norm] += 1
    
    return '\n'.join(result)


def filter_low_confidence_lines(text: str, ocr_data: Optional[Dict] = None, min_confidence: float = 25.0) -> str:
    """
    Filters out lines with average OCR confidence below threshold.
    
    Args:
        text: Input text
        ocr_data: Optional dict with 'tesseract_tsv' containing confidence scores
        min_confidence: Minimum average confidence (0-100) to keep a line
    
    Returns:
        Filtered text with low-confidence lines removed
    """
    # If no OCR data provided, apply heuristic filtering
    if not ocr_data or 'tesseract_tsv' not in ocr_data:
        return _heuristic_confidence_filter(text)
    
    # Extract line-level confidence from Tesseract TSV data
    tsv_rows = ocr_data.get('tesseract_tsv', [])
    if not tsv_rows:
        return text
    
    # Group words by line number (if available) or approximate by vertical position
    lines_text = text.split('\n')
    # For simplicity, assume each TSV row with text corresponds to sequential output
    # This is a simplified implementation; full implementation would need line tracking
    
    # Calculate average confidence for entire text
    confidences = [row.get('conf', 0) for row in tsv_rows if row.get('conf', -1) >= 0]
    if not confidences:
        return text
    
    avg_conf = sum(confidences) / len(confidences)
    
    # If overall confidence is very low, apply aggressive filtering
    if avg_conf < min_confidence:
        return _heuristic_confidence_filter(text)
    
    return text


def _heuristic_confidence_filter(text: str) -> str:
    """
    Heuristic filtering for low-quality OCR lines based on patterns.
    
    Removes lines that likely indicate poor OCR:
    - Very high ratio of non-alphanumeric characters
    - Random uppercase clusters (XBHAO, OJOHHEY, etc.)
    - Lines with repeated patterns suggesting noise
    """
    lines = text.split('\n')
    filtered = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append(line)
            continue
        
        # Calculate alphanumeric ratio
        alpha_count = sum(c.isalnum() for c in stripped)
        total_chars = len(stripped)
        
        if total_chars == 0:
            filtered.append(line)
            continue
        
        alpha_ratio = alpha_count / total_chars
        
        # Filter out lines with very low alphanumeric ratio
        if alpha_ratio < 0.4 and total_chars > 10:
            continue
        
        # Filter out lines with excessive uppercase clusters (likely noise)
        uppercase_clusters = re.findall(r'[A-Z]{6,}', stripped)
        if len(uppercase_clusters) > 2:
            continue
        
        # Filter out lines with repeated short tokens (OOO, XXX, etc.)
        tokens = stripped.split()
        if tokens:
            repeated_tokens = [t for t, count in Counter(tokens).items() if count > 3 and len(t) <= 4]
            if len(repeated_tokens) > 2:
                continue
        
        filtered.append(line)
    
    return '\n'.join(filtered)


def normalize_text_lines(text: str) -> str:
    """
    Normalizes text lines by removing excessive whitespace and fixing common OCR issues.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    normalized = []
    
    for line in lines:
        # Strip excessive whitespace
        line = ' '.join(line.split())
        
        # Fix common OCR substitutions
        # O (letter) vs 0 (zero) in numeric contexts
        line = re.sub(r'\bO(\d)', r'0\1', line)
        line = re.sub(r'(\d)O\b', r'\g<1>0', line)
        
        # Fix comma/period confusion
        line = re.sub(r'(\d)\s*,\s*(\d{3})\b', r'\1\2', line)  # 1,000 → 1000
        
        normalized.append(line)
    
    return '\n'.join(normalized)


__all__ = [
    'clean_text',
    'extract_tables_as_text',
    'deduplicate_lines',
    'filter_low_confidence_lines',
    'normalize_text_lines',
]
