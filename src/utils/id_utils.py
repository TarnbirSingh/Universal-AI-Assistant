import hashlib

def generate_unique_id(text: str) -> str:
    """Generate a unique ID based on text content"""
    return hashlib.md5(text.encode()).hexdigest()[:8]
