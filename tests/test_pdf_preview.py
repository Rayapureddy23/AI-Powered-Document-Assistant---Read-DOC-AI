import os

from src.ui import build_pdf_preview_html, get_previewable_paths


def test_get_previewable_paths_returns_only_pdf_documents():
    paths = ["/tmp/one.pdf", "/tmp/two.txt", "/tmp/three.HTML", "/tmp/four.PDF"]

    result = get_previewable_paths(paths)

    assert result == ["/tmp/one.pdf", "/tmp/four.PDF"]


def test_build_pdf_preview_html_contains_pdf_data_uri(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    html = build_pdf_preview_html(str(pdf_path))

    assert "data:application/pdf;base64" in html
    assert "application/pdf" in html
