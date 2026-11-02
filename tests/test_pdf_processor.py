import pymupdf
import pytest

from src.pdf_processor import PDFProcessingError, clean_text, extract_pdf
from tests.conftest import make_pdf, make_scanned_pdf


def test_extracts_text_with_page_numbers(two_page_pdf):
    doc = extract_pdf(two_page_pdf, "paper.pdf")
    assert doc.doc_name == "paper.pdf"
    assert doc.page_count == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "RNA sequencing" in doc.pages[0].text
    assert "DESeq2" in doc.pages[1].text
    assert not doc.is_scanned
    assert doc.warnings == []


def test_doc_id_is_stable_content_hash(two_page_pdf):
    a = extract_pdf(two_page_pdf, "a.pdf")
    b = extract_pdf(two_page_pdf, "renamed.pdf")
    assert a.doc_id == b.doc_id and len(a.doc_id) == 64


def test_clean_text_dehyphenates_and_normalises():
    raw = "The effi-\ncient   method uses ﬁltering.\x00\n\n\nSecond   paragraph."
    assert clean_text(raw) == "The efficient method uses filtering.\n\nSecond paragraph."


def test_clean_text_keeps_real_hyphens():
    assert clean_text("RNA-seq and single-cell data") == "RNA-seq and single-cell data"


def test_empty_bytes_raise():
    with pytest.raises(PDFProcessingError, match="empty"):
        extract_pdf(b"", "empty.pdf")


def test_corrupt_file_raises():
    with pytest.raises(PDFProcessingError, match="not a readable PDF"):
        extract_pdf(b"this is definitely not a pdf" * 10, "bad.pdf")


def test_encrypted_pdf_raises(tmp_path):
    doc = pymupdf.open(stream=make_pdf(["secret text"]), filetype="pdf")
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    doc.close()
    with pytest.raises(PDFProcessingError, match="password"):
        extract_pdf(path.read_bytes(), "locked.pdf")


def test_scanned_pdf_is_flagged_not_raised():
    doc = extract_pdf(make_scanned_pdf(), "scan.pdf")
    assert doc.is_scanned
    assert any("OCR" in w for w in doc.warnings)
    assert doc.text_pages == []


def test_blank_pdf_reports_no_text():
    doc = extract_pdf(make_pdf(["", ""]), "blank.pdf")
    assert not doc.is_scanned
    assert any("No extractable text" in w for w in doc.warnings)


def test_bare_page_numbers_are_removed():
    doc = extract_pdf(make_pdf(["12"]), "num.pdf")
    assert doc.pages[0].text == ""
