import re
import string
from typing import List

arabic_diacritics = re.compile(
    """
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """,
    re.VERBOSE,
)

arabic_punctuations = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
punctuations_list = arabic_punctuations + string.punctuation
translator = str.maketrans("", "", punctuations_list)
unicode_chars = ["\u200c", "\u200e", "\u200f", "\u202c"]


def remove_problematic_unicode(text: str) -> str:
    for x in unicode_chars:
        text = text.replace(x, "")
    text = text.strip()
    return text


def normalize_arabic(text: str) -> str:
    text = re.sub("[إأآاٱ]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text: str) -> str:
    text = re.sub(arabic_diacritics, "", text)
    return text


def clean_name(text: str, start_token: str = "!", end_token: str = "?") -> List[str]:
    text = text.lower()
    text = " ".join(text.split())
    text = text.translate(translator)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_problematic_unicode(text)
    return [start_token + e.strip() + end_token for e in text.split(" ")]
