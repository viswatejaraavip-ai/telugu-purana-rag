"""Parse GRETIL plaintext Puranas into one record per verse, plus an adhyaya manifest.

Handles the two reference styles seen on GRETIL:
  bhp_01.01.004        skandha.adhyaya.verse     (Bhagavata)
  vip_1,1.5            amsa,adhyaya.verse        (Vishnu)
Speaker lines: either "bhp_01.01.006/0 ṛṣaya ūcuḥ" or a bare "śrīsūta uvāca" line.
Output: corpus/<key>.jsonl (verses), corpus/<key>_adhyayas.jsonl (manifest with all verses).
"""
import os, re, sys
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(__file__))
from common import SOURCES, OUT, write_jsonl, iast_to_telugu, iast_to_devanagari, sha256, norm, GRETIL_LICENSE

WORKS = {
    "bhagavata": dict(
        file="sa_bhAgavatapurANa.txt", abbr="bhp", prefix="bhp",
        work="Bhagavata Purana", work_te="శ్రీమద్భాగవత పురాణము",
        book="skandha", book_te="స్కంధము", url="https://gretil.sub.uni-goettingen.de/gretil/corpustei/sa_bhAgavatapurANa.xml",
        ref=re.compile(r"//\s*bhp_(\d+)\.(\d+)\.(\d+)([a-z]?)(\*?)\s*//")),
    "vishnu": dict(
        file="sa_viSNupurANa.txt", abbr="vip", prefix="vip",
        work="Vishnu Purana", work_te="విష్ణు పురాణము",
        book="amsa", book_te="అంశము", url="https://gretil.sub.uni-goettingen.de/gretil/corpustei/sa_viSNupurANa.xml",
        ref=re.compile(r"//\s*vip_(\d+),(\d+)\.(\d+)([a-z]?)(\*?)\s*//")),
}
# ---- generic GRETIL works (many reference styles). book=None means single-book work.
GENERIC = {
    "markandeya": dict(file="sa_mArkaNDeyapurANa1-93.txt", abbr="markp", prefix="markp", work="Markandeya Purana", work_te="మార్కండేయ పురాణము", book_te=None),
    "garuda":     dict(file="sa_garuDapurANa.txt", abbr="garp", prefix="garp", work="Garuda Purana", work_te="గరుడ పురాణము", book_te="ఖండము"),
    "narasimha":  dict(file="sa_narasiMhapurANa.txt", abbr="NsP", prefix="nsp", work="Narasimha Purana", work_te="నృసింహ పురాణము", book_te=None),
    "shiva":      dict(file="sa_zivapurANabooks-1-and-7.txt", abbr="śivp", prefix="sivp", work="Shiva Purana (samhitas 1 and 7)", work_te="శివ పురాణము", book_te="సంహిత"),
    "kurma":      dict(file="sa_kUrmapurANa.txt", abbr="kūrmp", prefix="kurmp", work="Kurma Purana", work_te="కూర్మ పురాణము", book_te="భాగము"),
    "vamana":     dict(file="sa_vAmanapurANa1-69.txt", abbr="vamp", prefix="vamp", work="Vamana Purana", work_te="వామన పురాణము", book_te=None),
    "vamana_saromahatmya": dict(file="sa_vAmanapurANasaromAhAtmya.txt", abbr="vampsm", prefix="vampsm", work="Vamana Purana, Saromahatmya", work_te="వామన పురాణము (సరోమాహాత్మ్యము)", book_te=None),
    "matsya":     dict(file="sa_matsyapurANa1-176.txt", abbr="MatsP", prefix="matsp", work="Matsya Purana (adhyayas 1-176)", work_te="మత్స్య పురాణము", book_te=None),
    "agni":       dict(file="sa_agnipurANa.txt", abbr="ap", prefix="ap", work="Agni Purana", work_te="అగ్ని పురాణము", book_te=None),
    "linga":      dict(file="sa_liGgapurANa1-108.txt", abbr="LiP", prefix="lip", work="Linga Purana (purva bhaga)", work_te="లింగ పురాణము", book_te="భాగము"),
    "brahma":     dict(file="sa_brahmapurANa-1-246.txt", abbr="BrP", prefix="brp", work="Brahma Purana", work_te="బ్రహ్మ పురాణము", book_te=None),
    "narada":     dict(file="sa_nAradapurANa.txt", abbr="narp", prefix="narp", work="Narada Purana", work_te="నారద పురాణము", book_te="భాగము"),
    "brahmanda":  dict(file="sa_brahmANDapurANa.txt", abbr="bndp", prefix="bndp", work="Brahmanda Purana", work_te="బ్రహ్మాండ పురాణము", book_te="భాగము"),
    "skanda_revakhanda": dict(file="sa_skandapurANa-revAkhaNDa-rks.txt", abbr="rks", prefix="rks", work="Skanda Purana, Reva Khanda", work_te="స్కాంద పురాణము (రేవా ఖండము)", book_te=None),
    "vayu_revakhanda": dict(file="sa_revAkhANDa-of-the-vAyupurANa-rkv.txt", abbr="rkv", prefix="rkv", work="Vayu Purana, Reva Khanda", work_te="వాయు పురాణము (రేవా ఖండము)", book_te=None),
}
CHAPTER_WORD = re.compile(r"\bchapter\s+\d+\b")


def generic_ref(abbr):
    # delimiters seen: "// X //", "// X" (line end), "/X/", "//X/".  Body: digits with . , ; optional half suffix ab/cd/a-z; optional *
    # also the prose form "||(NsP_18.8)" seen in Narasimha
    return re.compile(r"(?:/{1,2}|\|\|\s*\(|\()\s*" + re.escape(abbr) + r"_([0-9][0-9.,]*?)([a-z]{0,2})(\*?)\s*(?:/{1,2}|\)|$)", re.M)


def parse_body(body):
    """'1,1.1'->(1,1,1) ; '1.1'->(None,1,1) ; '7.1,4.11'->('7.1',4,11) ; '1.003'->(None,1,3)"""
    if "," in body:
        b, rest = body.split(",", 1); c, v = rest.split(".")
        return (int(b) if b.isdigit() else b), int(c), int(v)
    c, v = body.split(".")
    return None, int(c), int(v)


def build_generic(key):
    w = GENERIC[key]
    path = os.path.join(SOURCES, "gretil", w["file"])
    text = open(path, encoding="utf-8").read().split("# Text", 1)[1]
    text = CHAPTER_WORD.sub(" ", text)
    rx = generic_ref(w["abbr"])
    pieces = OrderedDict(); order = []
    pos = 0
    for m in rx.finditer(text):
        raw = text[pos:m.start()]; pos = m.end()
        try: b, c, v = parse_body(m.group(1))
        except Exception: continue
        k = (b, c, v)
        if k not in pieces: pieces[k] = []; order.append(k)
        pieces[k].append(raw)
    records = []
    for k in order:
        b, c, v = k
        sloka, speaker = clean_verse("\n".join(pieces[k]))
        if not sloka: continue
        bno = b if b is not None else 1
        rid = f"{w['prefix']}-{bno}-{c}-{v}"
        book_te = f"{w['book_te']} {bno}" if w["book_te"] else ""
        cite_te = f"{w['work_te']}, {book_te + ', ' if book_te else ''}అధ్యాయము {c}, శ్లోకము {v}"
        cite_en = f"{w['work']} {str(bno) + '.' if b is not None else ''}{c}.{v}"
        records.append(OrderedDict(
            id=rid, text_type="verse", work=w["work"], work_te=w["work_te"], edition="GRETIL e-text",
            book_no=bno, book_te=book_te, adhyaya_no=c, verse_no=v, verse_suffix=None,
            citation_te=cite_te, citation_en=cite_en,
            speaker_iast=speaker, speaker_te=iast_to_telugu(speaker) if speaker else None,
            sloka_iast=sloka, sloka_te_script=iast_to_telugu(sloka), sloka_devanagari=iast_to_devanagari(sloka),
            gretil_flag_star=False, tatparyam_te=None, tatparyam_status=None,
            source="gretil", source_ref=f"{w['abbr']}_{m and ''}{'' if b is None else str(b)+','}{c}.{v}",
            source_url=f"https://gretil.sub.uni-goettingen.de/gretil/corpustei/{w['file'].replace('.txt', '.xml')}", license=GRETIL_LICENSE))
    n = write_jsonl(os.path.join(OUT, f"{key}.jsonl"), records)
    chapters = OrderedDict()
    for r in records: chapters.setdefault((r["book_no"], r["adhyaya_no"]), []).append(r)
    man = []
    for (bno, c), vs in chapters.items():
        book_te = vs[0]["book_te"]
        man.append(dict(adhyaya_id=f"{w['prefix']}-{bno}-{c}", work=w["work"], work_te=w["work_te"], book_no=bno, adhyaya_no=c,
                        n_verses=len(vs), citation_te=f"{w['work_te']}, {book_te + ', ' if book_te else ''}అధ్యాయము {c}",
                        verse_ids=[r["id"] for r in vs], speakers=sorted({r["speaker_iast"] for r in vs if r["speaker_iast"]})))
    write_jsonl(os.path.join(OUT, f"{key}_adhyayas.jsonl"), man)
    books = {}
    for cm in man: books[cm["book_no"]] = books.get(cm["book_no"], 0) + 1
    print(f"{key:22s}: {n:6d} verses, {len(man):4d} adhyayas, books={books}")
    return records


SPEAKER_TAG = re.compile(r"^\s*[a-z]+_[\d.,]+/0\s+(.+?)\s*$", re.M)
BARE_SPEAKER = re.compile(r"^\s*([^\n/]{2,60}?\s(?:uvāca|ūcuḥ|ūcatuḥ))\s*$", re.M)
MARKERS = re.compile(r"[\$&%\\]")


def clean_verse(raw):
    speakers = [m.group(1).strip() for m in SPEAKER_TAG.finditer(raw)]
    raw = SPEAKER_TAG.sub("", raw)
    speakers += [m.group(1).strip() for m in BARE_SPEAKER.finditer(raw)]
    raw = BARE_SPEAKER.sub("", raw)
    raw = MARKERS.sub(" ", raw)
    lines = [re.sub(r"\s+", " ", l).strip(" /") for l in raw.split("\n")]
    lines = [l for l in lines if l]
    return " |\n".join(lines), (speakers[-1] if speakers else None)


def build(key):
    w = WORKS[key]
    path = os.path.join(SOURCES, "gretil", w["file"])
    text = open(path, encoding="utf-8").read()
    text = text.split("# Text", 1)[1]
    records, seen = [], set()
    pos = 0
    for m in w["ref"].finditer(text):
        raw = text[pos:m.start()]; pos = m.end()
        b, a, v = int(m.group(1)), int(m.group(2)), int(m.group(3))
        suffix, star = m.group(4), m.group(5)
        rid = f"{w['prefix']}-{b}-{a}-{v}{suffix}"
        if rid in seen:            # GRETIL occasionally repeats a ref; keep the first
            continue
        seen.add(rid)
        sloka, speaker = clean_verse(raw)
        if not sloka:
            continue
        records.append(OrderedDict(
            id=rid, text_type="verse", work=w["work"], work_te=w["work_te"],
            edition="GRETIL e-text (Ulrich Stiehl input)",
            book_no=b, book_te=f"{w['book_te']} {b}", adhyaya_no=a, verse_no=v, verse_suffix=suffix or None,
            citation_te=f"{w['work_te']} {b}.{a}.{v}{suffix}",
            citation_en=f"{w['work']} {b}.{a}.{v}{suffix}",
            speaker_iast=speaker, speaker_te=iast_to_telugu(speaker) if speaker else None,
            sloka_iast=sloka, sloka_te_script=iast_to_telugu(sloka), sloka_devanagari=iast_to_devanagari(sloka),
            gretil_flag_star=bool(star), tatparyam_te=None, tatparyam_status=None,
            source="gretil", source_ref=f"{w['abbr']}_{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}",
            source_url=w["url"], license=GRETIL_LICENSE))
    n = write_jsonl(os.path.join(OUT, f"{key}.jsonl"), records)
    # adhyaya manifest
    chapters = OrderedDict()
    for r in records:
        chapters.setdefault((r["book_no"], r["adhyaya_no"]), []).append(r)
    man = []
    for (b, a), vs in chapters.items():
        man.append(dict(adhyaya_id=f"{w['prefix']}-{b}-{a}", work=w["work"], work_te=w["work_te"],
                        book_no=b, adhyaya_no=a, n_verses=len(vs),
                        citation_te=f"{w['work_te']}, {w['book_te']} {b}, అధ్యాయము {a}",
                        verse_ids=[r["id"] for r in vs],
                        speakers=sorted({r["speaker_iast"] for r in vs if r["speaker_iast"]})))
    write_jsonl(os.path.join(OUT, f"{key}_adhyayas.jsonl"), man)
    books = {}
    for c in man: books[c["book_no"]] = books.get(c["book_no"], 0) + 1
    print(f"{key}: {n} verses, {len(man)} adhyayas, books={books}, sha256={sha256(path)[:12]}")
    return records


if __name__ == "__main__":
    keys = sys.argv[1:] or list(WORKS) + list(GENERIC)
    for k in keys:
        if k in WORKS: build(k)
        else: build_generic(k)
