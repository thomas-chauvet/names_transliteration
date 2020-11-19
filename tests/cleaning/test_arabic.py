from names_transliteration.cleaning.arabic import (
    remove_problematic_unicode,
    normalize_arabic,
    remove_diacritics,
    clean_name,
)


def test_remove_problematic_unicode():
    assert remove_problematic_unicode("\u200c") == ""
    assert remove_problematic_unicode("\u200e") == ""
    assert remove_problematic_unicode("\u200f") == ""
    assert remove_problematic_unicode("\u202c") == ""
    assert remove_problematic_unicode("\u0600\u202c") == "\u0600"
    assert remove_problematic_unicode("\u0600\u202cabcdef") == "\u0600abcdef"


def test_normalize_arabic():
    assert normalize_arabic("إأآاٱ") == "ااااا"
    assert normalize_arabic("ىAbc") == "يAbc"
    assert normalize_arabic("ؤAbc") == "ءAbc"
    assert normalize_arabic("ئAbc") == "ءAbc"
    assert normalize_arabic("ةAbc") == "هAbc"
    assert normalize_arabic("گAbc") == "كAbc"


def test_remove_diacritics():
    assert remove_diacritics("إِعْجَام") == "إعجام"
    assert remove_diacritics("تَشْكِيل") == "تشكيل"
    assert remove_diacritics("حَرَكَات") == "حركات"
    assert remove_diacritics("حَرَكَة") == "حركة"


def test_clean_name():
    assert clean_name("إِعْجَام") == ['!اعجام?']
    assert clean_name("تَشْكِيل") == ["!تشكيل?"]
    assert clean_name("حَرَكَات") == ["!حركات?"]
    assert clean_name("حَرَكَة") == ["!حركه?"]
