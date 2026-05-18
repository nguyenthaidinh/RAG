#!/usr/bin/env python3
"""
R2 Smoke Test — Kiểm chứng extraction trực tiếp với file CTĐT thật.

Chạy:
    python scripts/smoke_extract_ctdt.py path/to/file.docx
    python scripts/smoke_extract_ctdt.py path/to/file.xlsx
    python scripts/smoke_extract_ctdt.py path/to/file.csv

Không cần DB, không cần FastAPI, không cần upload.
Gọi trực tiếp extract_text_with_metadata() và in kết quả.

Ví dụ output kỳ vọng:

  DOCX có bảng:
    - Marker [TABLE 1], [TABLE 2]... xuất hiện trong text
    - Paragraphs giữ nguyên thứ tự
    - Bảng hiển thị dạng pipe-delimited markdown

  XLSX:
    - Marker [SHEET: TênSheet] cho mỗi sheet có dữ liệu
    - Mỗi sheet là một bảng markdown
    - Sheet rỗng bị bỏ qua

  CSV:
    - Bảng markdown từ nội dung CSV
    - Encoding tiếng Việt hiển thị đúng
"""
from __future__ import annotations

import os
import sys
import time

# ── Ensure project root on sys.path ──────────────────────────────────

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main() -> None:
    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("  python scripts/smoke_extract_ctdt.py <file_path>")
        print()
        print("Hỗ trợ: .docx, .xlsx, .csv, .txt, .md, .pdf")
        print()
        print("Ví dụ:")
        print("  python scripts/smoke_extract_ctdt.py 'Mau07_CTDT.docx'")
        print("  python scripts/smoke_extract_ctdt.py 'DanhMuc_HocPhan.xlsx'")
        print("  python scripts/smoke_extract_ctdt.py 'KhaoSat_SV.csv'")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"❌ File không tồn tại: {file_path}")
        sys.exit(1)

    # ── Read file ─────────────────────────────────────────────────
    file_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)

    print("=" * 70)
    print(f"📄 File:      {filename}")
    print(f"📁 Path:      {file_path}")
    print(f"💾 Size:      {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print("=" * 70)

    with open(file_path, "rb") as f:
        data = f.read()

    # ── Extract ───────────────────────────────────────────────────
    from app.services.document_extract import (
        TableLimits,
        extract_text_with_metadata,
    )

    t0 = time.monotonic()
    try:
        text, meta = extract_text_with_metadata(
            filename,
            None,  # auto-detect from extension
            data,
            table_limits=TableLimits(max_rows=500, max_cols=50),
        )
    except Exception as e:
        print(f"\n❌ Extraction thất bại: {type(e).__name__}: {e}")
        sys.exit(1)

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    # ── Results ───────────────────────────────────────────────────
    text_len = len(text)
    has_table_marker = "[TABLE " in text
    has_sheet_marker = "[SHEET: " in text

    # Count markers
    table_markers = text.count("[TABLE ")
    sheet_markers = text.count("[SHEET: ")

    print()
    print("── Extraction Metadata ─────────────────────────────────────")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print()
    print("── Statistics ──────────────────────────────────────────────")
    print(f"  ⏱️  Thời gian extract:   {elapsed_ms} ms")
    print(f"  📝 Tổng ký tự text:     {text_len:,}")
    print(f"  📊 Markers [TABLE]:     {table_markers}")
    print(f"  📋 Markers [SHEET]:     {sheet_markers}")
    print(f"  🔍 has_table_marker:    {has_table_marker}")
    print(f"  🔍 has_sheet_marker:    {has_sheet_marker}")

    # ── Preview ───────────────────────────────────────────────────
    preview_len = min(2000, text_len)
    print()
    print(f"── Preview (đầu {preview_len} ký tự) ──────────────────────")
    print(text[:preview_len])

    if text_len > preview_len:
        print(f"\n... (còn {text_len - preview_len:,} ký tự)")

    # ── Table/Sheet sections ──────────────────────────────────────
    if has_table_marker or has_sheet_marker:
        print()
        print("── Markers tìm thấy ────────────────────────────────────")
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("[TABLE ") or stripped.startswith("[SHEET: "):
                print(f"  ✅ {stripped}")

    # ── Quality check hints ───────────────────────────────────────
    print()
    print("── Kiểm tra chất lượng ─────────────────────────────────────")

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".docx":
        if has_table_marker:
            print("  ✅ DOCX có bảng → markers [TABLE] đã extract")
        else:
            print("  ⚠️  DOCX không tìm thấy [TABLE] marker.")
            print("     Nếu file có bảng, kiểm tra lại extractor.")
            print("     Nếu file chỉ có paragraph → đúng hành vi.")
        if text_len < 50:
            print("  ⚠️  Text quá ngắn. File có thể rỗng hoặc chỉ có hình ảnh.")

    elif ext == ".xlsx":
        if has_sheet_marker:
            print("  ✅ XLSX → markers [SHEET] đã extract")
        else:
            print("  ⚠️  XLSX không tìm thấy [SHEET] marker.")
            print("     Có thể tất cả sheet đều rỗng.")

    elif ext == ".csv":
        if text_len > 0 and "|" in text:
            print("  ✅ CSV → markdown table đã extract")
        elif text_len == 0:
            print("  ⚠️  CSV rỗng.")
        else:
            print("  ⚠️  CSV có text nhưng không thấy pipe (|). Kiểm tra encoding.")

    # Kiểm tra cột CTĐT phổ biến
    ctdt_keywords = ["Mã HP", "Tên học phần", "Tín chỉ", "LT", "TH",
                     "Học kỳ", "Mục tiêu", "Chuẩn đầu ra", "CĐR",
                     "Tiên quyết", "STT", "BB", "TC"]
    found_keywords = [kw for kw in ctdt_keywords if kw in text]
    if found_keywords:
        print(f"  ✅ Từ khóa CTĐT tìm thấy: {', '.join(found_keywords)}")
    else:
        print("  ℹ️  Không tìm thấy từ khóa CTĐT đặc trưng (bình thường nếu file không phải tài liệu CTĐT).")

    print()
    print("=" * 70)
    print("✅ Smoke test hoàn tất.")
    print("=" * 70)


if __name__ == "__main__":
    main()
