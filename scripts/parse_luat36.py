"""Parse Luật 36/2024/QH15 PDF into pipeline-compatible JSON."""
import json, re, pdfplumber
from pathlib import Path

PDF_PATH = "data/raw/36-2024-qh15.pdf"
OUT_PATH = "data/raw/luat_36_2024.json"


def extract_full_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def parse_to_json(text: str) -> dict:
    chapters = []
    current_chapter = None
    current_article = None

    # Split into lines for processing
    lines = text.split("\n")

    chapter_pat = re.compile(r"^Chương\s+(.+)$")
    article_pat = re.compile(r"^Điều\s+(\d+)\.\s*(.+)$")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip header/footer lines
        if "CÔNG BÁO" in line or "QUỐC HỘI" in line or "Độc lập" in line:
            continue
        if line.startswith("Luật số:") or line == "LUẬT":
            continue
        if re.match(r"^\d+$", line):  # page numbers
            continue

        # Chapter detection
        chap_match = chapter_pat.match(line)
        if chap_match:
            chap_title = chap_match.group(1).strip()
            # Roman numeral to number
            roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
                         "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
            parts = chap_title.split(None, 1)
            roman = parts[0].rstrip(".")
            chap_num = roman_map.get(roman, len(chapters) + 1)
            chap_name = parts[1] if len(parts) > 1 else chap_title
            current_chapter = {
                "chapter_number": chap_num,
                "chapter_title": chap_name,
                "articles": [],
            }
            chapters.append(current_chapter)
            current_article = None
            continue

        # Article detection
        art_match = article_pat.match(line)
        if art_match:
            art_num = int(art_match.group(1))
            art_title = art_match.group(2).strip()
            current_article = {
                "article_number": art_num,
                "article_title": art_title,
                "content": "",
            }
            if current_chapter is None:
                current_chapter = {
                    "chapter_number": 1,
                    "chapter_title": "Unknown",
                    "articles": [],
                }
                chapters.append(current_chapter)
            current_chapter["articles"].append(current_article)
            continue

        # Accumulate article content
        if current_article is not None:
            if current_article["content"]:
                current_article["content"] += " "
            current_article["content"] += line

    # Post-process: clean up content
    for ch in chapters:
        for art in ch["articles"]:
            art["content"] = art["content"].strip()

    return {"chapters": chapters}


def main():
    print("Extracting text from PDF...")
    text = extract_full_text(PDF_PATH)

    print(f"Parsing {len(text)} chars into JSON...")
    data = parse_to_json(text)

    total_articles = sum(len(ch["articles"]) for ch in data["chapters"])
    print(f"Found {len(data['chapters'])} chapters, {total_articles} articles")

    # Print article 2 for verification
    for ch in data["chapters"]:
        for art in ch["articles"]:
            if art["article_number"] == 2:
                print(f"\nĐiều 2: {art['article_title']}")
                print(art["content"][:300])
                break

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
