"""Utility functions for pattern extraction and analysis"""
import re
from typing import Optional, List, Dict
from urllib.parse import urlparse

def extract_url_from_description(desc: str) -> Optional[str]:
    """Extract URL from step description."""
    # Try to find explicit URL
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    match = re.search(url_pattern, desc)
    if match:
        return match.group()
    
    # Try to find domain references
    domain_pattern = r'(?:go to|open|navigate to|visit)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    match = re.search(domain_pattern, desc.lower())
    if match:
        domain = match.group(1)
        # Add https:// if not present
        if not domain.startswith(('http://', 'https://')):
            return f"https://{domain}"
        return domain
    return None

def extract_search_term(desc: str) -> Optional[str]:
    """Extract search term from step description."""
    # Look for terms after search-related words
    patterns = [
        r'(?:search for|find|look for)\s+["\']([^"\']+)["\']',  # Quoted terms
        r'(?:search for|find|look for)\s+(.+?)(?:\s+in|\s+on|\s+at|$)',  # Unquoted terms
    ]
    
    for pattern in patterns:
        match = re.search(pattern, desc.lower())
        if match:
            return match.group(1).strip()
    return None

def extract_analysis_target(desc: str) -> List[str]:
    """Extract analysis targets from step description."""
    # Common data points to extract
    data_points = [
        'name', 'title', 'price', 'cost', 'rating', 'review',
        'description', 'summary', 'detail', 'specification',
        'feature', 'fact', 'information', 'data',
        'headquarters', 'location', 'address',
        'contact', 'phone', 'email',
        'industry', 'sector', 'category',
        'revenue', 'sales', 'profit',
        'employees', 'staff', 'team',
        'product', 'service', 'offering'
    ]
    
    found_points = []
    desc_lower = desc.lower()
    
    # Look for explicit mentions of data points
    for point in data_points:
        if point in desc_lower:
            found_points.append(point)
            
    # Check for additional context clues
    if 'key' in desc_lower:
        found_points.append('key_information')
    if 'main' in desc_lower:
        found_points.append('main_points')
    if 'important' in desc_lower:
        found_points.append('important_details')
        
    return found_points if found_points else ['general_information']

def extract_element(desc: str) -> Optional[str]:
    """Extract interactive element from step description."""
    # Look for elements in quotes
    quote_pattern = r'["\']([^"\']+)["\']'
    match = re.search(quote_pattern, desc)
    if match:
        return match.group(1)
    
    # Look for elements after action words
    element_pattern = r'(?:click|select|choose|press)\s+(?:the\s+)?([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)*?)(?:\s+button|\s+link|\s+tab|\s+menu|\s+option|$)'
    match = re.search(element_pattern, desc.lower())
    if match:
        return match.group(1)
    return None

def extract_field(desc: str) -> Optional[str]:
    """Extract input field from step description."""
    # Look for field references
    patterns = [
        r'(?:in|into|to)\s+(?:the\s+)?["\']([^"\']+)["\'](?:\s+field|\s+input|\s+box|$)',  # Quoted fields
        r'(?:in|into|to)\s+(?:the\s+)?([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)*?)(?:\s+field|\s+input|\s+box|$)'  # Unquoted fields
    ]
    
    for pattern in patterns:
        match = re.search(pattern, desc.lower())
        if match:
            return match.group(1)
    return None

def extract_input_value(desc: str) -> Optional[str]:
    """Extract input value from step description."""
    # Look for values in quotes
    quote_pattern = r'["\']([^"\']+)["\']'
    match = re.search(quote_pattern, desc)
    if match:
        return match.group(1)
        
    # Look for values after type/enter/input
    value_pattern = r'(?:type|enter|input)\s+([^"\'\s][^"\']+?)(?:\s+in|\s+into|\s+to|$)'
    match = re.search(value_pattern, desc.lower())
    if match:
        return match.group(1)
    return None

def enrich_step_context(step_type: str, desc: str) -> Dict:
    """Create rich context for a step based on its type and description."""
    context = {
        "action_type": step_type,
        "target_type": None,
        "parameters": {}
    }
    
    if step_type == "navigation":
        url = extract_url_from_description(desc)
        context.update({
            "target_type": "webpage",
            "parameters": {"url": url} if url else {}
        })
        
    elif step_type == "search":
        term = extract_search_term(desc)
        context.update({
            "target_type": "query",
            "parameters": {"term": term} if term else {}
        })
        
    elif step_type == "analysis":
        targets = extract_analysis_target(desc)
        context.update({
            "target_type": "content",
            "parameters": {"focus": targets} if targets else []
        })
        
    elif step_type == "interaction":
        element = extract_element(desc)
        context.update({
            "target_type": "element",
            "parameters": {"element": element} if element else {}
        })
        
    elif step_type == "input":
        field = extract_field(desc)
        value = extract_input_value(desc)
        context.update({
            "target_type": "field",
            "parameters": {
                "field": field if field else "",
                "value": value if value else ""
            }
        })
        
    return context
