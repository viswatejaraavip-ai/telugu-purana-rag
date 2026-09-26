"""Sanskrit Wikisource (CC BY-SA 4.0) -> verse records, same schema as build_gretil_purana.py.

Why: the GRETIL e-texts are CC BY-NC-SA (non-commercial). The chapter cards are
edition-independent (one Telugu card per adhyaya), so only the Sanskrit verse layer
is rebuilt here; ids keep the same shape (bhp-<skandha>-<adhyaya>-<verse>) so the
cards' verse_ids/anchors resolve against the new text where the two editions agree
on numbering, and the reconcile step reports where they do not.

    python scripts/build_wikisource_purana.py fetch  [work ...]   # cache wikitext under sources/wikisource/
    python scripts/build_wikisource_purana.py build  [work ...]   # corpus_ws/<work>.jsonl + _adhyayas.jsonl
    python scripts/build_wikisource_purana.py reconcile [work ...] # per-adhyaya verse counts vs corpus/

Works: bhagavata, vishnu, markandeya.
"""
import hashlib, json, os, re, sys, time, unicodedata, urllib.parse, urllib.request
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "sources", "wikisource")
OUT = os.path.join(ROOT, "corpus_ws")
OLD = os.path.join(ROOT, "corpus")
API = "https://sa.wikisource.org/w/api.php"
UA = {"User-Agent": "telugu-purana-rag corpus builder/0.1 (contact: viswa@tejaswiservices.com)"}
LICENSE = ("Sanskrit Wikisource, CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). "
           "Commercial use permitted with attribution and share-alike on the text.")

try:
    from indic_transliteration import sanscript
except ImportError:  # fetch/reconcile do not need it
    sanscript = None

WORKS = {
    "bhagavata": dict(prefix="bhp", page_prefix="श्रीमद्भागवतपुराणम्/", work="Bhagavata Purana",
                      work_te="శ్రీమద్భాగవత పురాణము", book_te="స్కంధము", cite="dotted"),
    "vishnu": dict(prefix="vip", page_prefix="विष्णुपुराणम्/", work="Vishnu Purana",
                   work_te="విష్ణు పురాణము", book_te="అంశము", cite="dotted"),
    "markandeya": dict(prefix="markp", page_prefix="मार्कण्डेयपुराणम्/", work="Markandeya Purana",
                       work_te="మార్కండేయ పురాణము", book_te="", cite="long"),
}
AMSHA = {"प्रथमांशः": 1, "द्वितीयांशः": 2, "तृतीयांशः": 3, "चतुर्थांशः": 4, "पञ्चमांशः": 5, "षष्टांशः": 6, "षष्ठांशः": 6}
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm(s):
    return unicodedata.normalize("NFC", s or "").replace("‌", "").replace("‍", "").strip()


def num(s):
    return int(norm(s).translate(DEV_DIGITS))


# ------------------------------------------------------------------ fetch

def api(params):
    params = dict(params, format="json")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=60))


def list_pages(page_prefix):
    pages, cont = [], {}
    while True:
        d = api(dict(action="query", list="allpages", apprefix=page_prefix, aplimit=500, apnamespace=0, **cont))
        pages += [p["title"] for p in d["query"]["allpages"]]
        if "continue" not in d:
            return pages
        cont = {"apcontinue": d["continue"]["apcontinue"]}


def slug(title):
    # Devanagari titles percent-encode past the filename limit; the title itself is
    # kept on the first line of the cached file.
    return hashlib.sha1(norm(title).encode("utf-8")).hexdigest()[:16] + ".txt"


def fetch(key):
    w = WORKS[key]
    d = os.path.join(SRC, key)
    os.makedirs(d, exist_ok=True)
    titles = list_pages(w["page_prefix"])
    json.dump(titles, open(os.path.join(d, "_pages.json"), "w"), ensure_ascii=False, indent=0)
    got = 0
    for t in titles:
        p = os.path.join(d, slug(t))
        if os.path.exists(p):
            continue
        r = api(dict(action="query", prop="revisions", rvprop="content|ids|timestamp", rvslots="main", titles=t))
        page = list(r["query"]["pages"].values())[0]
        rev = (page.get("revisions") or [{}])[0]
        text = ((rev.get("slots") or {}).get("main") or {}).get("*", "")
        meta = {"title": t, "revid": rev.get("revid"), "timestamp": rev.get("timestamp"),
                "url": "https://sa.wikisource.org/wiki/" + urllib.parse.quote(t.replace(" ", "_"))}
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n" + text)
        got += 1
        time.sleep(0.25)
    print(f"{key}: {len(titles)} pages, {got} fetched")


def load_page(key, title):
    with open(os.path.join(SRC, key, slug(title)), encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
        return meta, fh.read()


# ------------------------------------------------------------------ parse

SPEAKER = re.compile(r"^\(?([ऀ-ॿ][ऀ-ॿ\s\-]*?(?:उवाच|ऊचुः|उवाचः))\)?\s*[।॥]*\s*$")
VERSE_END = re.compile(r"[।॥]\s*([०-९0-9]+)\s*[।॥]+")           # ॥ २३ ॥ , । १ ॥ , ॥२३॥
VERSE_END_TAGGED = re.compile(r"॥\s*([ऀ-ॿ]+?)\s*([०-९0-9]+)\s*॥")  # ॥मंगल१॥
COLOPHON = re.compile(r"^\s*इति\s+श्री")
DROP_LINE = re.compile(r"^\s*(\{\{|\}\}|\||\[\[|<|\(|'''?\s*$|=+)")


def clean_wikitext(text):
    text = norm(text)
    text = re.sub(r"\{\{[Hh]eader.*?^\s*\}\}", "", text, flags=re.S | re.M)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</?(poem|div|span|center|small|big|b|i|u|p)[^>]*>", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\[\[[a-z\-]+:[^\]]*\]\]", "", text)               # interwiki
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"&nbsp;", " ", text)
    return text


# Verse-end markers as the editions on Wikisource write them:
#   ॥ २३ ॥   । १ ॥   ॥२३॥          plain
#   ॥१  ॥ २                          unclosed (Vishnu, several amshas)
#   ॥६.१॥   ॥ १,१५.१ ॥   ४-६ ।।      chapter- or book-prefixed (Markandeya, Vishnu)
#   ॥मंगल१॥                         tagged (invocation verses)
#   ...जनोऽयम् १।   ...महामुने १ ।     number then danda (Bhagavata 6-7, 10-11; Vishnu)
#   ...पराभवः  १                     bare number at line end (Bhagavata skandha 5)
_NUM = r"[०-९0-9]+"
_LET = r"[ऀ-॥॰-ॿ]"          # Devanagari letters and signs, not digits
_SEP = r"\s*[.,\-]\s*"
_BOOK = r"(?:%s%s(?=%s%s%s))?" % (_NUM, _SEP, _NUM, _SEP, _NUM)   # a leading book number only when chapter.verse follow
MARKER = re.compile(
    r"(?:[।॥]\s*(?:(?P<tag>%s+?)\s*)?%s(?:(?P<c>%s)%s)?(?P<n>%s)\s*[।॥]*"
    r"|(?<![।॥%s])\s%s(?:(?P<c2>%s)%s)?(?P<n2>%s)\s*[।॥]+)"
    % (_LET, _BOOK, _NUM, _SEP, _NUM, _LET, _BOOK, _NUM, _SEP, _NUM))   # book.chapter.verse, chapter.verse or verse
BARE_END = re.compile(r"\s%s(?:(?P<c>%s)[.\-])?(?P<n>%s)\s*$" % (_BOOK, _NUM, _NUM))
# A few pages carry a Hindi gloss between the verses; Sanskrit verse lines never use these words.
HINDI = re.compile(r"(?:^|\s)(?:है|हैं|वाली|वाला|वाले|करना|कराने|करने|बनाना|बनाने|जानना|और|में|लिए|द्वारा|आदि|यह|इस|उस|जो|तक)(?=\s|,|।|$)|,\s*$")
ADHYAYA_LINE = re.compile(r"ऽध्यायः|अध्यायः")


def split_verses(text):
    """(chapter_or_None, verse_no, suffix, text, speaker) from cleaned chapter text.

    Verse numbers are the markers in the text itself, so a chapter's numbering is
    whatever the edition says; when the marker carries a chapter prefix (॥६.१॥) it
    is returned so multi-chapter pages can be split on it. A line that is only a
    speaker tag attaches to the verses that follow. Colophons and chapter headings
    (anything naming an adhyaya) are skipped, never counted."""
    speaker, buf, out = None, [], []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or DROP_LINE.match(line):
            continue
        if COLOPHON.match(line) or (ADHYAYA_LINE.search(line) and len(line) < 120) or HINDI.search(line):
            buf = []
            continue
        m = SPEAKER.match(line)
        if m and len(line) < 60:
            speaker = m.group(1).strip()
            continue
        pos, hits = 0, list(MARKER.finditer(line))
        for m in hits:
            tag = m.group("tag")
            c = m.group("c") or m.group("c2")
            n = num(m.group("n") or m.group("n2"))
            body = " ".join(b for b in buf + [line[pos:m.start()].strip()] if b).strip(" ।॥")
            buf, pos = [], m.end()
            if body:
                out.append((num(c) if c else None, n, ("-" + tag) if tag else None,
                            re.sub(r"\s+", " ", body), speaker))
        rest = line[pos:].strip()
        if not hits:
            b = BARE_END.search(line)
            if b:
                body = " ".join(x for x in buf + [line[:b.start()].strip()] if x).strip(" ।॥")
                buf = []
                if body:
                    out.append((num(b.group("c")) if b.group("c") else None, num(b.group("n")), None,
                                re.sub(r"\s+", " ", body), speaker))
                continue
        if rest:
            buf.append(rest)
    return out


def chapter_pages(key):
    """(book_no, adhyaya_no, title) for every chapter page; duplicates resolved by
    preferring the canonical spelling and, failing that, the longer text."""
    titles = json.load(open(os.path.join(SRC, key, "_pages.json")))
    found = {}
    for t in titles:
        tt = norm(t)
        if key == "bhagavata":
            m = re.search(r"/(स्कन्धः|स्कन्दः|स्कंध)\s*([०-९]+)(?:/(पूर्वार्धः|उत्तरार्धः))?(?:/अध्यायः)?/अध्यायः\s*([०-९]+)$", tt)
            if not m:
                continue
            book, adh = num(m.group(2)), num(m.group(4))
            rank = 0 if m.group(1) == "स्कन्धः" else 1
        elif key == "vishnu":
            m = re.search(r"/([ऀ-ॿ]+ांशः)/अध्यायः\s*([०-९]+)$", tt)
            if not m or m.group(1) not in AMSHA:
                continue
            book, adh, rank = AMSHA[m.group(1)], num(m.group(2)), 0
        else:
            continue
        _, txt = load_page(key, t)
        cur = found.get((book, adh))
        cand = (rank, -len(txt), t)
        if cur is None or cand < cur:
            found[(book, adh)] = cand
    return sorted((b, a, v[2]) for (b, a), v in found.items())


MK_CHAPTER = re.compile(r"^\s*(\d+)\((\d+)\)\s*$")
MK_ORD = re.compile(r"([ऀ-ॿ]+?)ऽध्यायः\s*$")


def markandeya_chapters():
    """Markandeya pages hold 1-10 chapters each, in overlapping groupings. Every verse
    marker there carries its chapter (॥८१.१॥), so verses are grouped by that; pages
    whose markers lack it fall back to the 'N(M)' chapter line (N is Wikisource's own
    numbering, the one the cards follow) or the page title. Longest text per chapter wins."""
    titles = json.load(open(os.path.join(SRC, "markandeya", "_pages.json")))
    best = {}
    for t in titles:
        if "अध्याय" not in t:
            continue
        meta, txt = load_page("markandeya", t)
        txt = clean_wikitext(txt)
        m = re.search(r"अध्यायः\s*([०-९]+)\s*$", norm(t))
        cur = num(m.group(1)) if m else None
        groups = defaultdict(list)
        for line in txt.split("\n"):
            mk = MK_CHAPTER.match(line)
            if mk:
                cur = int(mk.group(1))
                continue
            for c, n, suffix, sa, speaker in split_verses(line):
                ch = c or cur
                if ch:
                    groups[ch].append((n, suffix, sa, speaker))
        for ch, vs in groups.items():
            if ch not in best or len(vs) > len(best[ch][1]):
                best[ch] = (t, vs)
    return [(1, ch, best[ch][0], best[ch][1]) for ch in sorted(best)]


def record(w, key, book, adh, vno, suffix, sa, speaker, url):
    dev = sa
    iast = sanscript.transliterate(dev, sanscript.DEVANAGARI, sanscript.IAST) if sanscript else ""
    te = sanscript.transliterate(dev, sanscript.DEVANAGARI, sanscript.TELUGU) if sanscript else ""
    sp_iast = sanscript.transliterate(speaker, sanscript.DEVANAGARI, sanscript.IAST) if (speaker and sanscript) else None
    sp_te = sanscript.transliterate(speaker, sanscript.DEVANAGARI, sanscript.TELUGU) if (speaker and sanscript) else None
    sfx = suffix or ""
    rid = f"{w['prefix']}-{book}-{adh}-{vno}{sfx}"
    book_te = f"{w['book_te']} {book}" if w["book_te"] else ""
    if w["cite"] == "dotted":
        cite_te = f"{w['work_te']} {book}.{adh}.{vno}{sfx}"
        cite_en = f"{w['work']} {book}.{adh}.{vno}{sfx}"
    else:
        cite_te = f"{w['work_te']}, అధ్యాయము {adh}, శ్లోకము {vno}{sfx}"
        cite_en = f"{w['work']} {adh}.{vno}{sfx}"
    return dict(id=rid, text_type="verse", work=w["work"], work_te=w["work_te"],
                edition="Sanskrit Wikisource (sa.wikisource.org)", book_no=book, book_te=book_te,
                adhyaya_no=adh, verse_no=vno, verse_suffix=suffix, citation_te=cite_te, citation_en=cite_en,
                speaker_iast=sp_iast, speaker_te=sp_te, sloka_iast=iast, sloka_te_script=te,
                sloka_devanagari=dev, gretil_flag_star=False, tatparyam_te=None, tatparyam_status=None,
                source="wikisource", source_ref=f"{w['prefix']}_{book}.{adh}.{vno}{sfx}", source_url=url,
                license=LICENSE)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def build(key):
    w = WORKS[key]
    if key == "markandeya":
        chapters = markandeya_chapters()
    else:
        chapters = []
        for book, adh, title in chapter_pages(key):
            meta, txt = load_page(key, title)
            chapters.append((book, adh, title, [v[1:] for v in split_verses(clean_wikitext(txt))]))
    records, man, empty = [], [], []
    for book, adh, title, verses in chapters:
        url = "https://sa.wikisource.org/wiki/" + urllib.parse.quote(norm(title).replace(" ", "_"))
        if not verses:
            empty.append((book, adh, title))
            continue
        # Some editions number each half-verse (… N । / … ॥N॥): consecutive entries
        # with the same number are one verse. A number that comes back later with
        # the same text is a pasted duplicate and is dropped; with different text it
        # is kept under a -r2 suffix so nothing silently vanishes.
        # A page may hold the chapter twice, from two editions: the numbering
        # restarts. Keep the longest single run of numbering.
        # restarts. A run ends only on a true restart (back to the first verses
        # after the numbering has run on), never on a mistyped number. The first
        # run wins unless a later one is clearly the fuller text.
        # A lone dip (…14, 1, 16…) is a mistyped digit and is renumbered in place.
        fixed = []
        for i, v in enumerate(verses):
            nxt = verses[i + 1][0] if i + 1 < len(verses) else None
            if fixed and v[0] < fixed[-1][0] and nxt == fixed[-1][0] + 2:
                v = (fixed[-1][0] + 1,) + tuple(v[1:])
            fixed.append(v)
        verses = fixed
        runs, prev = [[]], 0
        for i, v in enumerate(verses):
            nxt = verses[i + 1][0] if i + 1 < len(verses) else None
            if v[0] <= 3 and prev >= 10 and runs[-1] and nxt is not None and nxt <= v[0] + 2:
                runs.append([])
            runs[-1].append(v)
            prev = v[0]
        verses = runs[0]
        for r in runs[1:]:
            if len(r) >= 1.5 * len(verses):
                verses = r
        merged = []
        for vno, suffix, sa, speaker in verses:
            if merged and merged[-1][0] == vno and merged[-1][1] == suffix:
                merged[-1][2] += " " + sa
            else:
                merged.append([vno, suffix, sa, speaker])
        seen, vs = {}, []
        for vno, suffix, sa, speaker in merged:
            rid = f"{w['prefix']}-{book}-{adh}-{vno}{suffix or ''}"
            if rid in seen:
                if seen[rid] == sa:
                    continue
                k = 2
                while f"{rid}-r{k}" in seen:
                    k += 1
                suffix = (suffix or "") + f"-r{k}"
                rid = f"{rid}-r{k}"
            seen[rid] = sa
            vs.append(record(w, key, book, adh, vno, suffix, sa, speaker, url))
        records += vs
        book_te = f"{w['book_te']} {book}" if w["book_te"] else ""
        man.append(dict(adhyaya_id=f"{w['prefix']}-{book}-{adh}", work=w["work"], work_te=w["work_te"],
                        book_no=book, adhyaya_no=adh, n_verses=len(vs),
                        citation_te=f"{w['work_te']}, {book_te + ', ' if book_te else ''}అధ్యాయము {adh}",
                        verse_ids=[r["id"] for r in vs],
                        speakers=sorted({r["speaker_iast"] for r in vs if r["speaker_iast"]}),
                        source_url=url))
    n = write_jsonl(os.path.join(OUT, f"{key}.jsonl"), records)
    write_jsonl(os.path.join(OUT, f"{key}_adhyayas.jsonl"), man)
    print(f"{key:12s}: {n:6d} verses, {len(man):4d} adhyayas; {len(empty)} chapter pages with no verses parsed")
    for e in empty[:10]:
        print("   empty:", e)


# ------------------------------------------------------------------ reconcile

def reconcile(key):
    old = {r["adhyaya_id"]: r for r in map(json.loads, open(os.path.join(OLD, f"{key}_adhyayas.jsonl"), encoding="utf-8"))}
    new = {r["adhyaya_id"]: r for r in map(json.loads, open(os.path.join(OUT, f"{key}_adhyayas.jsonl"), encoding="utf-8"))}
    same = close = far = 0
    missing_new = [a for a in old if a not in new]
    extra_new = [a for a in new if a not in old]
    old_ids = {vid for r in old.values() for vid in r["verse_ids"]}
    new_ids = {vid for r in new.values() for vid in r["verse_ids"]}
    diffs = []
    for a, r in old.items():
        if a not in new:
            continue
        d = int(new[a]["n_verses"]) - int(r["n_verses"])
        if d == 0:
            same += 1
        elif abs(d) <= 2:
            close += 1
        else:
            far += 1
            diffs.append((a, int(r["n_verses"]), int(new[a]["n_verses"])))
    print(f"{key:12s}: cards {len(old)} | chapters found {len(new)} | verse count same {same}, ±2 {close}, "
          f"off by >2 {far} | card verse ids resolving in new text {len(old_ids & new_ids)}/{len(old_ids)} "
          f"| cards without a chapter {len(missing_new)} | new chapters without a card {len(extra_new)}")
    if missing_new:
        print("   no chapter for:", missing_new[:12])
    if extra_new:
        print("   no card for:", extra_new[:12], "..." if len(extra_new) > 12 else "")
    for a, o, n in sorted(diffs, key=lambda x: -abs(x[1] - x[2]))[:12]:
        print(f"   {a}: gretil {o} verses, wikisource {n}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    keys = sys.argv[2:] or list(WORKS)
    for k in keys:
        {"fetch": fetch, "build": build, "reconcile": reconcile}[cmd](k)
