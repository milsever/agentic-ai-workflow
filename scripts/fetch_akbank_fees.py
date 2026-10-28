"""Fetch Akbank fee tables and write only banking_products/akbank_fees.jsonl.

Run from the repository root:
    python scripts/fetch_akbank_fees.py

Dependencies: requests, beautifulsoup4
"""

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SOURCE_URL = "https://www.akbank.com/urun-ve-hizmet-ucretleri"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "documents" / "banking_products" / "akbank_fees.jsonl"
TIMEOUT_SECONDS = 30
USER_AGENT = "Mozilla/5.0 (compatible; EnterpriseRAGPublicSourceCollector/1.0)"

# Explicit section-to-category mapping established for this project.
# The source page's category markup is inconsistent, so category is always
# assigned from this mapping rather than inferred from HTML structure.
SECTION_CATEGORY_FALLBACK = {
    "Havale Ücreti": "PARA VE KIYMETLI MADEN TRANSFERI",
    "Elektronik Fon Transferi EFT Ücreti": "PARA VE KIYMETLI MADEN TRANSFERI",
    "Uluslararası Fon Transferi ve Mesajlaşma Ücreti": "PARA VE KIYMETLI MADEN TRANSFERI",
    "Kıymetli Maden Transferi Ücreti": "PARA VE KIYMETLI MADEN TRANSFERI",
    "Akbank Hesap Özeti": "MEVDUAT HESAPLARı",
    "Mevduat Hesapları": "MEVDUAT HESAPLARı",
    "Para Çekme Ücreti": "MEVDUAT HESAPLARı",
    "Kart ve Hizmet Ücretleri": "KREDI KARTLARı VE BANKA KARTLARı",
    "Bireysel Kredi Kartları Faiz Oranları": "KREDI KARTLARı VE BANKA KARTLARı",
    "Konut Kredileri": "BIREYSEL KREDILER",
    "Taşıt Kredileri": "BIREYSEL KREDILER",
    "İhtiyaç Kredileri": "BIREYSEL KREDILER",
    "Çekler ve Senetler": "ÇEKLER VE SENETLER",
    "Kiralık Kasa Ücretleri": "KIRALıK KASA ÜCRETLERI",
    "Menkul Kıymet İşlemleri": "MENKUL KıYMET İŞLEMLERI",
    "Başka Kuruluş ATMsinden Yapılan İşlem Ücreti": "DIĞER",
    "Diğer": "DIĞER",
    "Kampanyalı Ürün veya Hizmetler Ücreti": "DIĞER",
    "Kredi Kartları": "KOBI KREDILERI",
    "Ortak Ücretler": "KOBI KREDILERI",
    "Gayrinakit Krediler": "KOBI KREDILERI",
    "Nakit Krediler": "KOBI KREDILERI",
    "Tarım Kart": "KOBI KREDILERI",
    "Ürün ve Hizmet Paketleri": "KOBI KREDILERI",
}

def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()

def section_before(table) -> str:
    """Use the closest non-empty heading as the table section label."""
    for heading in table.find_all_previous(["h1", "h2", "h3", "h4", "h5"]):
        text = clean_text(heading.get_text(" ", strip=True))
        if text:
            return text
    return "Ürün ve Hizmet Ücretleri"

def extract_records(soup: BeautifulSoup) -> list[dict]:
    records: list[dict] = []
    for table_index, table in enumerate(soup.find_all("table"), start=1):
        section = section_before(table)
        category = SECTION_CATEGORY_FALLBACK.get(section, "DIĞER")
        rows = table.find_all("tr")
        if not rows:
            continue

        header_cells = rows[0].find_all(["th", "td"], recursive=False)
        headers = [clean_text(cell.get_text(" ", strip=True)) for cell in header_cells]

        for row_index, row in enumerate(rows[1:], start=1):
            cells = row.find_all(["th", "td"], recursive=False)
            values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
            if not any(values):
                continue

            # Keep the output schema compact: no `fields` object.
            record = {
                "category": category,
                "source_url": SOURCE_URL,
                "section": section,
                "table_index": table_index,
                "row_index": row_index,
                "headers": headers,
                "values": values,
            }
            records.append(record)
    return records

def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    response = session.get(SOURCE_URL, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    if "text/html" not in response.headers.get("Content-Type", "").lower():
        raise RuntimeError("Akbank kaynağı beklenen HTML içeriğini döndürmedi.")

    soup = BeautifulSoup(response.text, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()

    records = extract_records(soup)
    if not records:
        raise RuntimeError(
            "Sayfadan tablo satırı çıkarılamadı; mevcut JSONL dosyası değiştirilmedi."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # This script creates/updates only the requested JSONL output file.
    temp_path = OUTPUT_PATH.with_suffix(".jsonl.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path.replace(OUTPUT_PATH)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    unmapped_sections = sorted({
        record["section"] for record in records
        if record["section"] not in SECTION_CATEGORY_FALLBACK
    })
    print(f"Kaynak: {SOURCE_URL}")
    print(f"Çıkarılan kayıt: {len(records)}")
    print(f"JSONL: {OUTPUT_PATH}")
    if unmapped_sections:
        print("Eşlemede bulunmayan section değerleri 'DIĞER' olarak kaydedildi:")
        for section in unmapped_sections:
            print(f"  - {section}")

if __name__ == "__main__":
    main()
