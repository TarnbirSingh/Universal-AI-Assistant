import re

def escape_markdown_v2(text: str) -> str:
    """
    Helper function to escape special characters for MARKDOWN_V2 while preserving
    existing markdown formatting.
    Must escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
    """
    # First, temporarily replace valid markdown patterns
    placeholders = {
        'bold': (r'\*\*(.+?)\*\*', '‡B‡\g<1>‡B‡'),  # Bold
        'italic': (r'\*(.+?)\*', '‡I‡\g<1>‡I‡'),    # Italic
        'code': (r'`(.+?)`', '‡C‡\g<1>‡C‡'),        # Code
        'pre': (r'```(.+?)```', '‡P‡\g<1>‡P‡')      # Pre-formatted
    }
    
    # Save existing markdown
    for pattern, placeholder in placeholders.values():
        text = re.sub(pattern, placeholder, text)
    
    # Escape special characters
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    
    # Restore markdown
    text = text.replace('‡B‡', '*')  # Bold
    text = text.replace('‡I‡', '_')  # Italic
    text = text.replace('‡C‡', '`')  # Code
    text = text.replace('‡P‡', '```') # Pre-formatted
    
    return text