from finsent.nlp.cleaning import clean_text, text_hash


def test_clean_text_removes_noise() -> None:
    raw = "<p>Nvidia&nbsp;beats estimates</p> https://example.com/story [+42 chars] …   next"

    cleaned = clean_text(raw)

    assert cleaned == "Nvidia beats estimates next"


def test_clean_text_handles_empty_values() -> None:
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t ") == ""


def test_text_hash_is_stable_and_sensitive() -> None:
    first = text_hash("Nvidia beats estimates")
    second = text_hash("Nvidia beats estimates")
    other = text_hash("Nvidia misses estimates")

    assert first == second
    assert first != other
    assert len(first) == 64
