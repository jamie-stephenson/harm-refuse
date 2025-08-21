"""
Utilities for separating prompts based on whether
a model refuses to answer them.
"""
from nnsight import LanguageModel

refusal_phrases = [
    "Sorry",
    # TODO: Complete this. Should it be hardcoded, or in e.g. a .txt file?
]

def is_refused(
    response: str,
) -> bool:
    for phrase in refusal_phrases:
        if phrase in response:
            return True
    return False
