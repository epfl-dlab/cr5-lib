import typing
import nltk
import numpy as np


def tokenize_document(document: str) -> typing.List[str]:
    """
    Helper method to tokenize the document.
    :param document: The input document represented as a string.
    :return: A list of tokens.
    """
    try:
        return nltk.tokenize.word_tokenize(document)
    except LookupError:
        nltk.download('punkt')
        return nltk.tokenize.word_tokenize(document)


def is_valid_str(s: str) -> bool:
    """
    Check if the input string is not empty.
    :param s: Input string.
    :return: True if the string is not empty.
    """
    if s is None:
        return False
    if not isinstance(s, str):
        return False
    if len(s) <= 0:
        return False
    return True


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize the vector based on l-2 norm.
    :param vec: A numpy array.
    :return: The normalized numpy array.
    """
    return vec / np.linalg.norm(vec)

