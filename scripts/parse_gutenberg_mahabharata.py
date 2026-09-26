"""Ganguli's Mahabharata (1883-96) from Project Gutenberg #15474-15477 -> english/mahabharata.jsonl.

English Wikisource carries only the Adi Parva of it; the four Gutenberg volumes hold all
eighteen parvas. Sections follow the Calcutta (vulgate) adhyayas, so "BOOK n / SECTION m"
maps to mbh-<n>-<m>; where the Sanskrit corpus is a different recension the section
numbers drift and the English is offered to the card writer only for parvas whose
chapter counts agree (gen_epic_cards_batch.py --english-books). Footnote apparatus is
not in these files; "(Sanskrit gloss)" parentheticals are Ganguli's own and are kept.

usage: python scripts/parse_gutenberg_mahabharata.py pg15474.txt pg15475.txt pg15476.txt pg15477.txt
"""
import json, os, re, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "english", "mahabharata.jsonl")
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
PARVA = {1: "Adi", 2: "Sabha", 3: "Vana", 4: "Virata", 5: "Udyoga", 6: "Bhishma", 7: "Drona", 8: "Karna", 9: "Shalya",
         10: "Sauptika", 11: "Stri", 12: "Shanti", 13: "Anushasana", 14: "Ashvamedhika", 15: "Ashramavasika",
         16: "Mausala", 17: "Mahaprasthanika", 18: "Svargarohana"}


def roman(s):
    n = 0
    for i, ch in enumerate(s):
        v = ROMAN[ch]
        n += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return n


def main(paths):
    rows = []
    for path in paths:
        raw = open(path, encoding="utf-8-sig").read().replace("\r", "")
        body = raw.split("*** START OF", 1)[-1].split("*** END OF", 1)[0]
        book, sec, buf = None, None, []

        def flush():
            if book and sec and buf:
                text = re.sub(r"[ \t]+", " ", "\n".join(buf)).strip()
                text = re.sub(r"\n{3,}", "\n\n", text)
                if len(text) > 200:
                    rows.append(dict(adhyaya_id=f"mbh-{book}-{sec}", book_no=book, adhyaya_no=sec,
                                     title_en=f"{PARVA.get(book, '')} Parva, section {sec}", text_en=text,
                                     translator="Kisari Mohan Ganguli",
                                     source_title=f"The Mahabharata of Krishna-Dwaipayana Vyasa (Project Gutenberg #{os.path.basename(path)[2:7]})",
                                     source_url="https://www.gutenberg.org/ebooks/" + os.path.basename(path)[2:7],
                                     license="Public domain (translator died 1908); text from Project Gutenberg"))

        for l in body.split("\n"):
            s = l.strip()
            m = re.match(r"^BOOK (\d+)$", s)
            if m:
                flush(); book, sec, buf = int(m.group(1)), None, []
                continue
            m = re.match(r"^SECTION ([IVXLCDM]+)\.?$", s)
            if m and book:
                flush(); sec, buf = roman(m.group(1)), []
                continue
            # Books 8-11 and 16-18 mark a section with a bare number line; only the next
            # expected number counts, so a stray numeral inside the prose never splits a section.
            if book and re.match(r"^\d+$", s) and int(s) == (sec or 0) + 1:
                flush(); sec, buf = int(s), []
                continue
            if re.match(r"^END OF [A-Z ]+PARVA", s) or re.match(r"^THE END", s):
                flush(); sec = None
                continue
            if book and sec is not None:
                buf.append(l)
        flush()
    # a section that appears twice (a volume's table of contents repeats headings) keeps the longer text
    best = {}
    for r in rows:
        if r["adhyaya_id"] not in best or len(r["text_en"]) > len(best[r["adhyaya_id"]]["text_en"]):
            best[r["adhyaya_id"]] = r
    rows = sorted(best.values(), key=lambda r: (r["book_no"], r["adhyaya_no"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} sections; per parva {dict(sorted(collections.Counter(r['book_no'] for r in rows).items()))}; "
          f"median chars {sorted(len(r['text_en']) for r in rows)[len(rows) // 2]}")
    r = rows[0]; print(r["adhyaya_id"], "|", r["text_en"][:240].replace("\n", " / "))


if __name__ == "__main__":
    main(sys.argv[1:])
