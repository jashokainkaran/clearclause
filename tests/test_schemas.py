"""
Stage 0 input validation - enforced once, at the API boundary, via a
Pydantic field_validator on ProvisionRequest.text. FastAPI turns a
ValidationError into its own 422 response before any endpoint body runs,
so these tests exercise the validator directly rather than through HTTP.
"""

import unicodedata

import pytest
from pydantic import ValidationError

from backend.schemas import ProvisionRequest, MAX_INPUT_LENGTH_CHARS


def test_empty_text_is_rejected():
    with pytest.raises(ValidationError):
        ProvisionRequest(text="")


def test_whitespace_only_text_is_rejected():
    with pytest.raises(ValidationError):
        ProvisionRequest(text="   \n\t  ")


def test_valid_normal_input_is_accepted():
    req = ProvisionRequest(text="Whoever commits theft shall be punished with imprisonment.")
    assert req.text == "Whoever commits theft shall be punished with imprisonment."


def test_oversized_input_is_rejected():
    with pytest.raises(ValidationError):
        ProvisionRequest(text="a" * (MAX_INPUT_LENGTH_CHARS + 1))


def test_input_at_exact_maximum_is_accepted():
    text = "a" * MAX_INPUT_LENGTH_CHARS
    req = ProvisionRequest(text=text)
    assert len(req.text) == MAX_INPUT_LENGTH_CHARS


def test_unicode_is_normalised_to_nfc():
    # A single precomposed accented character (NFC) vs the same accented
    # character spelled as a base letter + a separate combining accent
    # (NFD) must normalise to the same precomposed form before
    # length/validity checks run, so behaviour is consistent regardless of
    # input encoding. Built via chr()/unicodedata rather than a literal
    # accented character in the source file, so this test is not sensitive
    # to the file's own on-disk encoding.
    accented = chr(0x00E9)  # LATIN SMALL LETTER E WITH ACUTE (precomposed)
    composed = "caf" + accented
    decomposed = "caf" + unicodedata.normalize("NFD", accented)
    assert composed != decomposed  # sanity check the fixture really differs

    req = ProvisionRequest(text=decomposed)
    assert req.text == composed
    assert unicodedata.is_normalized("NFC", req.text)
