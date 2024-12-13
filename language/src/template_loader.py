import re
from pathlib import Path


def clean_string(s: str) -> str:
    """Removes leading, trailing, and multiple spaces."""
    s = s.strip()
    s = re.sub("\s\s+", " ", s)

    return s


def get_template(file: str | Path) -> str:
    """Loads a template from a file."""
    with open(file, "r", encoding="utf8") as f:
        template = f.read()

    return clean_string(template)
