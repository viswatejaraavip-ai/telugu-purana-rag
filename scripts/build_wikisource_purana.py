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

def W(prefix, page_prefix, work, work_te, book_te="", cite="long", books=None, corpus=None, chapters=None,
      unmatched="keep", exclude_map=None):
    """books: title segment(s) between the page prefix and the chapter -> book_no (as the
    cards write it); None means the work has one book. corpus: corpus/ file key when it
    differs from the WORKS key (two GRETIL works can map to one Wikisource text)."""
    return dict(prefix=prefix, page_prefix=page_prefix, work=work, work_te=work_te, book_te=book_te,
                cite=cite, books=books, corpus=corpus, chapters=chapters, unmatched=unmatched, exclude_map=exclude_map)
    # chapters: (lo, hi) of ws chapter numbers this work takes. unmatched="drop": once aligned, keep only
    # chapters that map to a card (a work sharing another work's page tree). exclude_map: skip the ws
    # chapters that this other work's chapter map claimed.

REVA = "स्कन्दपुराणम्/खण्डः ५ (अवन्तीखण्डः)/रेवा खण्डम्/"
WORKS = {
    "bhagavata": W("bhp", "श्रीमद्भागवतपुराणम्/", "Bhagavata Purana", "శ్రీమద్భాగవత పురాణము", "స్కంధము", "dotted",
                   books=lambda seg: (lambda m: num(m.group(2)) if m else None)(
                       re.match(r"^(स्कन्धः|स्कन्दः|स्कंध)\s*([०-९]+)(?:/(?:पूर्वार्धः|उत्तरार्धः))?(?:/अध्यायः)?$", seg))),
    "vishnu": W("vip", "विष्णुपुराणम्/", "Vishnu Purana", "విష్ణు పురాణము", "అంశము", "dotted",
                books={"प्रथमांशः": 1, "द्वितीयांशः": 2, "तृतीयांशः": 3, "चतुर्थांशः": 4, "पञ्चमांशः": 5, "षष्टांशः": 6, "षष्ठांशः": 6}),
    "markandeya": W("markp", "मार्कण्डेयपुराणम्/", "Markandeya Purana", "మార్కండేయ పురాణము"),
    "agni": W("ap", "अग्निपुराणम्/", "Agni Purana", "అగ్ని పురాణము"),
    "brahma": W("brp", "ब्रह्मपुराणम्/", "Brahma Purana", "బ్రహ్మ పురాణము"),
    "brahmanda": W("bndp", "ब्रह्माण्डपुराणम्/", "Brahmanda Purana", "బ్రహ్మాండ పురాణము", "భాగము",
                   books={"पूर्वभागः": 1, "मध्यभागः": 2, "उत्तरभागः": 3}),
    "garuda": W("garp", "गरुडपुराणम्/", "Garuda Purana", "గరుడ పురాణము", "ఖండము",
                books={"आचारकाण्डः": 1, "प्रेतकाण्डः (धर्मकाण्डः)": 2, "ब्रह्मकाण्डः (मोक्षकाण्डः)": 3}),
    "kurma": W("kurmp", "कूर्मपुराणम्-", "Kurma Purana", "కూర్మ పురాణము", "భాగము",
               books={"पूर्वभागः": 1, "उत्तरभागः": 2}),
    "linga": W("lip", "लिङ्गपुराणम् - ", "Linga Purana", "లింగ పురాణము", "భాగము",
               books={"पूर्वभागः": 1, "उत्तरभागः": 2}),
    "matsya": W("matsp", "मत्स्यपुराणम्/", "Matsya Purana", "మత్స్య పురాణము"),
    "narada": W("narp", "नारदपुराणम्- ", "Narada Purana", "నారద పురాణము", "భాగము",
                books={"पूर्वार्धः": 1, "उत्तरार्धः": 2}),
    "shiva": W("sivp", "शिवपुराणम्/", "Shiva Purana", "శివ పురాణము", "సంహిత",
               books={"संहिता १ (विश्वेश्वरसंहिता)": "1",
                      "संहिता २ (रुद्रसंहिता)/खण्डः १ (सृष्टिखण्डः)": "2.1", "संहिता २ (रुद्रसंहिता)/खण्डः २ (सतीखण्डः)": "2.2",
                      "संहिता २ (रुद्रसंहिता)/खण्डः ३ (पार्वतीखण्डः)": "2.3", "संहिता २ (रुद्रसंहिता)/खण्डः ४ (कुमारखण्डः)": "2.4",
                      "संहिता २ (रुद्रसंहिता)/खण्डः ५ (युद्धखण्डः)": "2.5", "संहिता ३ (शतरुद्रसंहिता)": "3",
                      "संहिता ४ (कोटिरुद्रसंहिता)": "4", "संहिता ५ (उमासंहिता)": "5", "संहिता ६ (कैलाससंहिता)": "6",
                      "संहिता ७ (वायवीयसंहिता)/पूर्व भागः": "7.1", "संहिता ७ (वायवीयसंहिता)/उत्तर भागः": "7.2"}),
    # The Reva Khanda on Wikisource is one text; GRETIL carried two recensions (as Skanda
    # and as Vayu). Both are built from it and reconcile says which set of cards it matches.
    "vayu_revakhanda": W("rkv", REVA, "Vayu Purana, Reva Khanda", "వాయు పురాణము (రేవా ఖండము)"),
    # Wikisource files the Saromahatmya as the first 26 chapters of the Vamana Purana
    # Wikisource interleaves the Saromahatmya (chapters 22-48 of its tree) with the Vamana Purana proper;
    # content alignment sorts them out: build the Saromahatmya first, then Vamana without its chapters.
    "vamana_saromahatmya": W("vampsm", "वामनपुराणम्/", "Vamana Purana, Saromahatmya", "వామన పురాణము (సరోమాహాత్మ్యము)", unmatched="drop"),
    "vamana": W("vamp", "वामनपुराणम्/", "Vamana Purana", "వామన పురాణము", exclude_map="vamana_saromahatmya"),
    # not on Sanskrit Wikisource (2026-09): narasimha; the Skanda recension of the Reva Khanda -> cards only
}
_AMSHA_UNUSED = {"प्रथमांशः": 1, "द्वितीयांशः": 2, "तृतीयांशः": 3, "चतुर्थांशः": 4, "पञ्चमांशः": 5, "षष्टांशः": 6, "षष्ठांशः": 6}
# Typists mix digit blocks (a Kannada ೦ inside a Devanagari number is common), so
# every Indic decimal digit maps to ASCII.
DEV_DIGITS = str.maketrans("०१२३४५६७८९" "೦೧೨೩೪೫೬೭೮೯" "౦౧౨౩౪౫౬౭౮౯" "০১২৩৪৫৬৭৮৯" "૦૧૨૩૪૫૬૭૮૯" "൦൧൨൩൪൫൬൭൮൯" "௦௧௨௩௪௫௬௭௮௯",
                           "0123456789" * 7)
_DIG = r"०-९0-9೦-೯౦-౯০-৯૦-૯൦-൯௦-௯"


def norm(s):
    return unicodedata.normalize("NFC", s or "").replace("‌", "").replace("‍", "").strip()


def num(s):
    return int(norm(s).translate(DEV_DIGITS))


# ------------------------------------------------------------------ fetch

def api(params, tries=5):
    params = dict(params, format="json")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=UA)
    for k in range(tries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception as e:  # read timeouts on big pages, the odd 5xx
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


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


def src_dir(key):
    # works that share a Wikisource text share one cache
    pp = WORKS[key]["page_prefix"]
    return os.path.join(SRC, "reva" if pp == REVA else "vamana" if pp == "वामनपुराणम्/" else key)


def fetch(key):
    w = WORKS[key]
    d = src_dir(key)
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
    with open(os.path.join(src_dir(key), slug(title)), encoding="utf-8") as fh:
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
_NUM = r"[%s]+" % _DIG
_LET = r"[\u0900-\u0963\u0970-\u097F]"          # Devanagari letters and signs: not digits, not dandas
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


_ORD_SIMPLE = {"प्रथम": 1, "द्वितीय": 2, "तृतीय": 3, "चतुर्थ": 4, "पञ्चम": 5, "षष्ठ": 6, "षष्ट": 6, "सप्तम": 7,
               "अष्टम": 8, "नवम": 9, "दशम": 10, "एकादश": 11, "द्वादश": 12, "त्रयोदश": 13, "चतुर्दश": 14,
               "पञ्चदश": 15, "षोडश": 16, "सप्तदश": 17, "अष्टादश": 18, "शततम": 100}
_ORD_TENS = [("नवति", 90), ("णवति", 90), ("शीति", 80), ("सप्तति", 70), ("षष्टि", 60), ("षष्ठि", 60), ("पञ्चाशत्", 50), ("पञ्चाषत्", 50),
             ("चत्वारिंशत्", 40), ("चत्वारिंशात्", 40), ("त्रिंशत्", 30), ("विंशति", 20)]
_ORD_UNITS = [("एकोन", -1), ("ऊन", -1), ("एका", 1), ("एक", 1), ("द्व्य", 2), ("द्वा", 2), ("द्वि", 2), ("त्र्य", 3), ("त्रयस्", 3),
              ("त्रयो", 3), ("त्रि", 3), ("चतुश्", 4), ("चतुष्", 4), ("चतुस्", 4), ("चतुर्", 4), ("चतुर", 4), ("पञ्चा", 5),
              ("पञ्च", 5), ("पञ्ज", 5), ("षड्", 6), ("षट्", 6), ("षण्", 6), ("षडा", 6), ("षड", 6), ("सप्ता", 7), ("सप्त", 7),
              ("अष्टा", 8), ("अष्ट", 8), ("नवा", 9), ("नव", 9), ("अ", 0), ("", 0)]


def ordinal(word):
    """Sanskrit ordinal as written in a chapter title -> int, or None.
    Tens are matched by their stem so sandhi forms (एकाशीति, द्व्यशीति) resolve; the
    unit is whatever precedes the stem."""
    w = norm(word)
    for k, v in _ORD_SIMPLE.items():      # सप्तमोऽध्यायः holds "तमो" inside the stem: simple forms first
        if re.match(k + r"(ो|ः)?ऽ?ध्यायः", w):
            return v
    w = re.sub(r"(तमो|ो|ः)?ऽ?ध्यायः.*$", "", w).rstrip("ो")
    for tens, tv in _ORD_TENS:
        i = w.rfind(tens)
        if i < 0:
            continue
        head = w[:i]
        for unit, uv in _ORD_UNITS:
            if head.endswith(unit):
                return tv + uv
        return None
    for k, v in _ORD_SIMPLE.items():
        if w == k:
            return v
    return None


def chapter_pages(key):
    """(book_no, adhyaya_no, title) for every chapter page of a work. The page prefix
    is stripped, the segments before the last name the book (WORKS[key]['books']),
    and the last segment carries the chapter as a number or an ordinal word. When two
    pages claim one chapter, the plainer title wins, then the longer text."""
    w = WORKS[key]
    titles = json.load(open(os.path.join(src_dir(key), "_pages.json")))
    found = {}
    for t in titles:
        tt = norm(t)
        if not tt.startswith(norm(w["page_prefix"])):
            continue
        rest = tt[len(norm(w["page_prefix"])):]
        segs = rest.split("/")
        last, book_seg = segs[-1].strip(), "/".join(x.strip() for x in segs[:-1])
        if w["books"] is None:
            if book_seg:
                continue
            book = 1
        elif callable(w["books"]):
            book = w["books"](book_seg)
            if book is None:
                continue
        else:
            book = w["books"].get(book_seg)
            if book is None:
                continue
        m = re.match(r"^अध्याय(?:ाः|ः|:)?\s*-?\s*([%s]+)$" % _DIG, last)
        if m:
            adh, rank = num(m.group(1)), 0
        elif "ध्याय" in last:
            adh, rank = ordinal(last), 1
            if adh is None:
                print("   unparsed chapter title:", t)
                continue
        else:
            continue
        if w["chapters"] and not (w["chapters"][0] <= adh <= w["chapters"][1]):
            continue
        _, txt = load_page(key, t)
        cand = (rank, -len(txt), t)
        cur = found.get((book, adh))
        if cur is None or cand < cur:
            found[(book, adh)] = cand
    return sorted(((b, a, v[2]) for (b, a), v in found.items()), key=lambda x: (str(x[0]), x[1]))


def _chapter_pages_old(key):
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
        cite_te = f"{w['work_te']}, {book_te + ', ' if book_te else ''}అధ్యాయము {adh}, శ్లోకము {vno}{sfx}"
        cite_en = f"{w['work']} {str(book) + '.' if book_te else ''}{adh}.{vno}{sfx}"
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


def _apply_verse_map(w, key, vs, vmap):
    """Renumber a chapter's records to the card ids the verse map assigns; merge
    consecutive records that map to one id; keep unmapped records under their own
    number, suffixed -w when a mapped record now owns that id."""
    if not vmap:
        return vs
    from indic_transliteration import sanscript
    expanded = []
    for r in vs:
        m = vmap.get(r["id"])
        if not isinstance(m, dict):
            expanded.append(r)
            continue
        # cut the record at the recorded spans: each piece carries one card id, the
        # text before the first span (if any) stays with the record's own id
        text = r["sloka_devanagari"]
        parts = m["parts"]
        cuts = [0] + [(parts[k][2] + parts[k + 1][1]) // 2 for k in range(len(parts) - 1)] + [len(text)]
        lead = text[:parts[0][1]].strip(" ।॥") if m.get("own") and parts[0][1] > 12 else ""
        if lead:
            lid = r["id"] + "#lead"
            expanded.append(dict(r, id=lid, sloka_devanagari=lead,
                                 sloka_iast=sanscript.transliterate(lead, sanscript.DEVANAGARI, sanscript.IAST),
                                 sloka_te_script=sanscript.transliterate(lead, sanscript.DEVANAGARI, sanscript.TELUGU)))
            vmap[lid] = m["own"]
            cuts[0] = parts[0][1]
        for k, (cid, s0, s1) in enumerate(parts):
            piece = text[cuts[k]:cuts[k + 1]].strip(" ।॥")
            if not piece:
                continue
            expanded.append(dict(r, id=r["id"] + f"#{k}", sloka_devanagari=piece,
                                 sloka_iast=sanscript.transliterate(piece, sanscript.DEVANAGARI, sanscript.IAST),
                                 sloka_te_script=sanscript.transliterate(piece, sanscript.DEVANAGARI, sanscript.TELUGU)))
            vmap[r["id"] + f"#{k}"] = cid
    vs = expanded
    out = []
    for r in vs:
        target = vmap.get(r["id"])
        if target and out and out[-1]["id"] == target:
            for f in ("sloka_iast", "sloka_te_script", "sloka_devanagari"):
                out[-1][f] = (out[-1][f] + " " + r[f]).strip()
            continue
        if target:
            m = re.match(r"^%s-([^-]+)-(\d+)-(\d+)(-.*)?$" % re.escape(w["prefix"]), target)
            tb = m.group(1)
            r = dict(r, id=target, book_no=(int(tb) if tb.isdigit() else tb),
                     book_te=(f"{w['book_te']} {tb}" if w["book_te"] else ""),
                     adhyaya_no=int(m.group(2)), verse_no=int(m.group(3)), verse_suffix=(m.group(4) or None),
                     source_ref=r["source_ref"] + " (ws " + r["id"] + ")")
            sfx = m.group(4) or ""
            if w["cite"] == "dotted":
                r["citation_te"] = f"{w['work_te']} {r['book_no']}.{r['adhyaya_no']}.{r['verse_no']}{sfx}"
                r["citation_en"] = f"{w['work']} {r['book_no']}.{r['adhyaya_no']}.{r['verse_no']}{sfx}"
            else:
                bt = f"{r['book_te']}, " if r["book_te"] else ""
                r["citation_te"] = f"{w['work_te']}, {bt}అధ్యాయము {r['adhyaya_no']}, శ్లోకము {r['verse_no']}{sfx}"
                r["citation_en"] = f"{w['work']} {str(r['book_no']) + '.' if r['book_te'] else ''}{r['adhyaya_no']}.{r['verse_no']}{sfx}"
        out.append(r)
    owned = {r["id"] for r in out if r["source_ref"].endswith(")")}
    seen = set()
    final = []
    for r in out:
        if r["id"] in seen or (r["id"] in owned and not r["source_ref"].endswith(")")):
            r = dict(r, id=r["id"] + "-w", verse_suffix=(r["verse_suffix"] or "") + "-w")
        seen.add(r["id"])
        final.append(r)
    return final


def build(key):
    w = WORKS[key]
    if key == "markandeya":
        chapters = markandeya_chapters()
    else:
        chapters = []
        for book, adh, title in chapter_pages(key):
            meta, txt = load_page(key, title)
            chapters.append((book, adh, title, [v[1:] for v in split_verses(clean_wikitext(txt))]))
    # A chapter map from `align` renumbers chapters to the cards' numbering. An
    # unmapped chapter keeps its own number unless a mapped chapter now owns it,
    # in which case it is kept under "<n>w" so no text is lost.
    cmap_path = os.path.join(OUT, f"{key}_chapter_map.json")
    cmap = json.load(open(cmap_path, encoding="utf-8")) if os.path.exists(cmap_path) else {}
    owned = {v["to"] for v in cmap.values()}
    if w.get("exclude_map"):
        xp = os.path.join(OUT, f"{w['exclude_map']}_chapter_map.json")
        taken_by_other = set(json.load(open(xp, encoding="utf-8"))) if os.path.exists(xp) else set()
        chapters = [c for c in chapters if f"{c[0]}-{c[1]}" not in taken_by_other]
    if cmap and w.get("unmatched") == "drop":
        chapters = [c for c in chapters if f"{c[0]}-{c[1]}" in cmap]
    renumbered, parked = 0, []
    fixed = []
    for book, adh, title, verses in chapters:
        m = cmap.get(f"{book}-{adh}")
        if m:
            b2, a2 = m["to"].rsplit("-", 1)
            if (str(b2), int(a2)) != (str(book), adh):
                renumbered += 1
            book, adh = (int(b2) if str(b2).isdigit() else b2), int(a2)
        elif f"{book}-{adh}" in owned:
            parked.append(f"{book}-{adh}")
            adh = f"{adh}w"
        fixed.append((book, adh, title, verses))
    chapters = fixed
    if cmap:
        print(f"   chapter map applied: {renumbered} renumbered, {len(parked)} parked as <n>w: {parked[:8]}")
    vmap_path = os.path.join(OUT, f"{key}_verse_map.json")
    vmap = json.load(open(vmap_path, encoding="utf-8")) if os.path.exists(vmap_path) else {}
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
        # Typists drop the hundreds after verse 100 (…99, 100, 1, 2, …, 9, 110…):
        # a number that is exactly the expected one minus a multiple of 100 is the
        # expected one. A lone dip (…14, 1, 16…) is a mistyped digit, renumbered too.
        fixed = []
        for i, v in enumerate(verses):
            nxt = verses[i + 1][0] if i + 1 < len(verses) else None
            if fixed:
                exp = fixed[-1][0] + 1
                if v[0] < exp and exp >= 100 and (exp - v[0]) % 100 == 0:
                    v = (exp,) + tuple(v[1:])
                elif v[0] < fixed[-1][0] and nxt == fixed[-1][0] + 2:
                    v = (exp,) + tuple(v[1:])
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
        vs = _apply_verse_map(w, key, vs, vmap)
        records += vs
    groups = {}
    for r in records:
        groups.setdefault((r["book_no"], r["adhyaya_no"]), []).append(r)
    for (book, adh), vs in groups.items():
        book_te = f"{w['book_te']} {book}" if w["book_te"] else ""
        man.append(dict(adhyaya_id=f"{w['prefix']}-{book}-{adh}", work=w["work"], work_te=w["work_te"],
                        book_no=book, adhyaya_no=adh, n_verses=len(vs),
                        citation_te=f"{w['work_te']}, {book_te + ', ' if book_te else ''}అధ్యాయము {adh}",
                        verse_ids=[r["id"] for r in vs],
                        speakers=sorted({r["speaker_iast"] for r in vs if r["speaker_iast"]}),
                        source_url=vs[0]["source_url"]))
    n = write_jsonl(os.path.join(OUT, f"{key}.jsonl"), records)
    write_jsonl(os.path.join(OUT, f"{key}_adhyayas.jsonl"), man)
    print(f"{key:12s}: {n:6d} verses, {len(man):4d} adhyayas; {len(empty)} chapter pages with no verses parsed")
    for e in empty[:10]:
        print("   empty:", e)


# ------------------------------------------------------------------ align

def _grams(text, n=4):
    t = re.sub(r"[^a-zA-Zāīūṛṝḷṃḥśṣñṅṭḍṇ']", "", text.lower())
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def _chapter_text(rows, limit=1200):
    return " ".join(r["sloka_iast"] for r in rows)[:limit]


def _verse_overlap(nrows, ogs, t=0.22):
    """Fraction of Wikisource verses in a chapter that have a matching card verse."""
    hit = 0
    for r in nrows:
        g = _vgrams(r["sloka_iast"])
        if g and any(len(g & h) / len(g | h) >= t for h in ogs if h):
            hit += 1
    return hit / max(1, len(nrows))


def align(key, window=10, min_score=0.5):
    """Editions number chapters differently in places. Map every Wikisource chapter
    to the card chapter (GRETIL numbering) whose text it is, scored by the share of
    its verses that match a verse of the candidate, first within ±window in the
    same book, then work-wide for whatever is left on both sides. Writes
    corpus_ws/<key>_chapter_map.json {"<ws book>-<ws adhyaya>": {"to": ..., "score": s}}.
    The card corpus text is read for the comparison only; nothing of it is written out."""
    ck = WORKS[key]["corpus"] or key
    old_rows = defaultdict(list)
    for r in map(json.loads, open(os.path.join(OLD, f"{ck}.jsonl"), encoding="utf-8")):
        old_rows[(str(r["book_no"]), int(r["adhyaya_no"]))].append(r)
    new_rows = defaultdict(list)
    for r in map(json.loads, open(os.path.join(OUT, f"{key}.jsonl"), encoding="utf-8")):
        new_rows[(str(r["book_no"]), int(r["adhyaya_no"]))].append(r)
    old_g = {k: [_vgrams(o["sloka_iast"]) for o in v] for k, v in old_rows.items()}
    taken, shifts = {}, defaultdict(int)

    def claim(nk, ok, score):
        prev = taken.get(ok)
        if prev and prev[1] >= score:
            return False
        taken[ok] = (nk, score)
        return True

    for (book, adh), rows in sorted(new_rows.items(), key=lambda x: (x[0][0], x[0][1])):
        best, score = None, 0.0
        for d in range(-window, window + 1):
            cand = (book, adh + d)
            if cand in old_g:
                sc = _verse_overlap(rows, old_g[cand])
                if sc > score:
                    best, score = cand, sc
        if best and score >= min_score:
            claim((book, adh), best, score)
    # leftovers: a ws chapter nobody placed against a card chapter nobody claimed
    placed = {v[0] for v in taken.values()}
    for nk, rows in new_rows.items():
        if nk in placed:
            continue
        best, score = None, 0.0
        for ok in old_g:
            if ok in taken:
                continue
            sc = _verse_overlap(rows, old_g[ok])
            if sc > score:
                best, score = ok, sc
        if best and score >= min_score:
            claim(nk, best, score)
    out = {}
    for ok, (nk, score) in taken.items():
        out[f"{nk[0]}-{nk[1]}"] = {"to": f"{ok[0]}-{ok[1]}", "score": round(score, 3)}
        shifts[ok[1] - nk[1] if ok[0] == nk[0] else "book"] += 1
    unmatched = [f"{b}-{a}" for (b, a) in new_rows if f"{b}-{a}" not in out]
    json.dump(out, open(os.path.join(OUT, f"{key}_chapter_map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{key:12s}: {len(new_rows)} wikisource chapters, {len(out)} matched to a card chapter "
          f"(shift histogram {dict(sorted(shifts.items(), key=str))}), {len(unmatched)} unmatched: {unmatched[:12]}")


# ------------------------------------------------------------------ align verses

def _vgrams(iast, n=3):
    t = re.sub(r"[^a-zāīūṛṝḷṃḥśṣñṅṭḍṇ]", "", (iast or "").lower())
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def align_verses(key, min_score=0.2):
    """Within each chapter that both editions have (after the chapter map), match
    every Wikisource verse to the card-corpus verse it is, by 3-gram overlap of the
    IAST text, and record the id it should carry. Consecutive Wikisource verses that
    are both halves of one card verse map to the same id and are merged by build.
    Writes corpus_ws/<key>_verse_map.json {ws_id: card_id}. The card-corpus text
    is read for the comparison only."""
    ck = WORKS[key]["corpus"] or key
    old = defaultdict(list)
    for r in map(json.loads, open(os.path.join(OLD, f"{ck}.jsonl"), encoding="utf-8")):
        old[f"{r['book_no']}-{r['adhyaya_no']}"].append(r)
    new = defaultdict(list)
    for r in map(json.loads, open(os.path.join(OUT, f"{key}.jsonl"), encoding="utf-8")):
        new[f"{r['book_no']}-{r['adhyaya_no']}"].append(r)
    vmap, stats = {}, defaultdict(int)
    for ch, nrows in new.items():
        orows = old.get(ch)
        if not orows:
            stats["ws verses in chapters without a card"] += len(nrows)
            continue
        og = [(o["id"], _vgrams(o["sloka_iast"])) for o in orows]
        ng = [(r["id"], _vgrams(r["sloka_iast"])) for r in nrows]
        pairs = []
        for i, (nid, g) in enumerate(ng):
            if not g:
                continue
            for j, (oid, h) in enumerate(og):
                if not h:
                    continue
                inter = len(g & h)
                if inter:
                    pairs.append((inter / len(g | h), i, j))
        pairs.sort(reverse=True)
        used_n, used_o = set(), {}
        for score, i, j in pairs:
            if score < min_score or i in used_n:
                continue
            # a card verse may absorb two consecutive ws verses (half-verse numbering)
            if j in used_o and not (abs(used_o[j] - i) == 1 and score >= min_score):
                continue
            used_n.add(i)
            used_o.setdefault(j, i)
            vmap[ng[i][0]] = og[j][0]
        # Card verses still unplaced: an edition that cuts prose into more units than
        # Wikisource does. Find each unit's span inside the ws record it belongs to and
        # record a split; build cuts the record at those spans. Text stays Wikisource's.
        leftovers = [j for j in range(len(og)) if j not in used_o]
        if leftovers:
            import difflib
            from indic_transliteration import sanscript
            splits = defaultdict(list)     # ws index -> [(card_id, start, end)]
            for j in leftovers:
                unit = sanscript.transliterate(orows[j]["sloka_iast"], sanscript.IAST, sanscript.DEVANAGARI)
                unit = re.sub(r"[\s|।॥]+", " ", unit).strip()
                if len(unit) < 6:
                    continue
                best = None
                for i, r in enumerate(nrows):
                    text = r["sloka_devanagari"]
                    sm = difflib.SequenceMatcher(None, text, unit, autojunk=False)
                    blocks = [bl for bl in sm.get_matching_blocks() if bl.size >= 6]
                    if not blocks:
                        continue
                    cov = sum(bl.size for bl in blocks) / len(unit)
                    span = (blocks[0].a, blocks[-1].a + blocks[-1].size)
                    if cov >= 0.25 and (best is None or cov > best[0]):
                        best = (cov, i, span)
                if best:
                    splits[best[1]].append((og[j][0], best[2][0], best[2][1]))
                    used_o[j] = best[1]
            for i, parts in splits.items():
                parts.sort(key=lambda x: x[1])
                # a record that was itself matched keeps its id for the text outside the spans
                own = vmap.get(ng[i][0])
                vmap[ng[i][0]] = {"own": own, "parts": parts}
                stats["card verses placed by span"] += len(parts)
        stats["ws verses matched"] += len(used_n)
        stats["ws verses unmatched"] += len(ng) - len(used_n)
        stats["card verses matched"] += len(used_o)
        stats["card verses with no ws verse"] += len(og) - len(used_o)
    # Last pass: card verses still unplaced are looked for anywhere in the work
    # (a chapter filed under another number, a duplicate page). Strict threshold.
    placed = {v for v in vmap.values() if isinstance(v, str)} | {p[0] for v in vmap.values() if isinstance(v, dict) for p in v["parts"]}
    free = [(r["id"], _vgrams(r["sloka_iast"])) for rows in new.values() for r in rows if r["id"] not in vmap]
    moved = 0
    for ch, orows in old.items():
        for o in orows:
            if o["id"] in placed:
                continue
            g = _vgrams(o["sloka_iast"])
            if len(g) < 12:
                continue
            best, score = None, 0.0
            for nid, h in free:
                if not h:
                    continue
                sc = len(g & h) / len(g | h)
                if sc > score:
                    best, score = nid, sc
            if best and score >= 0.4:
                vmap[best] = o["id"]
                placed.add(o["id"])
                free = [(n, h) for n, h in free if n != best]
                moved += 1
    stats["card verses found elsewhere in the work"] = moved
    json.dump(vmap, open(os.path.join(OUT, f"{key}_verse_map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{key:12s}: verse alignment {dict(stats)}")


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


def pipeline(key):
    """build on Wikisource numbering, align to the cards, rebuild, reconcile."""
    for f in (f"{key}_chapter_map.json", f"{key}_verse_map.json"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            os.remove(p)
    build(key)
    align(key)
    build(key)
    align_verses(key)
    build(key)
    reconcile(key)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    keys = sys.argv[2:] or list(WORKS)
    for k in keys:
        {"fetch": fetch, "build": build, "reconcile": reconcile, "align": align, "verses": align_verses, "all": pipeline}[cmd](k)
