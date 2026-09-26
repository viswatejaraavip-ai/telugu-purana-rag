"""Griffith's Ramayana (1870-74) from Project Gutenberg #24869 -> english/ramayana.jsonl.

English Wikisource carries only Books I-III of it; Gutenberg has Books I-VI complete
(Griffith summarised the Uttara Kanda in prose, which is not aligned to sargas and is
left out). Canto numbers follow Valmiki's sargas. Footnote markers "(12)" and the
footnote apparatus are dropped; the verse lines are kept as lines.

usage: python scripts/parse_gutenberg_ramayana.py <pg24869.txt>
"""
import json, os, re, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "english", "ramayana.jsonl")
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman(s):
    n = 0
    for i, ch in enumerate(s):
        v = ROMAN[ch]
        n += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return n


def main(path):
    raw = open(path, encoding="utf-8-sig").read().replace("\r", "")
    body = raw.split("*** START OF", 1)[-1].split("*** END OF", 1)[0]
    # the table of contents repeats the BOOK/Canto lines; the text proper starts at the first
    # "BOOK I." that is followed by "Canto I." within a few lines
    lines = body.split("\n")
    start = next(i for i, l in enumerate(lines)
                 if re.match(r"^BOOK I\.", l.strip()) and any(re.match(r"^Canto I\.", x.strip()) for x in lines[i:i + 8]))
    lines = lines[start:]
    book, canto, title, buf, rows = None, None, None, [], []

    def flush():
        if book and canto and buf:
            text = "\n".join(buf).strip()
            text = re.sub(r"\(\d+\)", "", text)                       # footnote markers
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > 200:
                rows.append(dict(adhyaya_id=f"ram-{book}-{canto}", book_no=book, adhyaya_no=canto, title_en=title,
                                 text_en=text, translator="R. T. H. Griffith",
                                 source_title="The Rámáyan of Válmíki (Project Gutenberg #24869)",
                                 source_url="https://www.gutenberg.org/ebooks/24869",
                                 license="Public domain (translator died 1906); text from Project Gutenberg"))

    for l in lines:
        s = l.strip()
        m = re.match(r"^BOOK ([IVX]+)\.", s)
        if m and roman(m.group(1)) == (book or 0) + 1:
            flush(); book, canto, buf = roman(m.group(1)), None, []
            continue
        m = re.match(r"^Canto ([IVXLC]+)\.\s*(.*)$", s)
        if m and book:
            flush(); canto = roman(m.group(1)); title = re.sub(r"\(\d+\)", "", m.group(2)).strip(" ."); buf = []
            continue
        if re.match(r"^FOOTNOTES", s) or re.match(r"^APPENDIX", s):
            flush(); book = None
            continue
        if book and canto is not None:
            buf.append(l)
    flush()
    rows.sort(key=lambda r: (r["book_no"], r["adhyaya_no"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} cantos; per book {dict(collections.Counter(r['book_no'] for r in rows))}; "
          f"median chars {sorted(len(r['text_en']) for r in rows)[len(rows) // 2]}")
    r = rows[0]; print(r["adhyaya_id"], r["title_en"], "|", r["text_en"][:300].replace("\n", " / "))


if __name__ == "__main__":
    main(sys.argv[1])
