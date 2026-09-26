"""Public-domain English translations from English Wikisource, one record per chapter,
aligned to the Sanskrit corpus ids -- the epics' meaning layer, at no model cost.

  Ramayana: R. T. H. Griffith (1870-74), "The Ramayana/Book I/Canto I: Nárad"  -> ram-<kanda>-<sarga>
  Mahabharata: K. M. Ganguli (1883-96), "The Mahabharata/Book 1: Adi Parva/Section 1" -> mbh-<parva>-<section>

Writes english/<work>.jsonl: {adhyaya_id, book_no, adhyaya_no, title_en, text_en, source_url, license}.
Griffith's canto numbers follow Valmiki's sargas but he abridged parts (notably the
Uttara Kanda); a sarga with no canto simply has no English record.

usage: python scripts/fetch_english.py ramayana|mahabharata
"""
import json, os, re, sys, time, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "english")
API = "https://en.wikisource.org/w/api.php"
UA = {"User-Agent": "telugu-purana-rag corpus builder/0.1 (contact: viswa@tejaswiservices.com)"}
LICENSE = "Public domain (translator died before 1926); text from English Wikisource"

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman(s):
    s = s.upper().strip(); n = 0
    for i, ch in enumerate(s):
        v = ROMAN[ch]
        n += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return n


WORKS = {
    "ramayana": dict(prefix="ram", page_prefix="The Ramayana/",
                     title=re.compile(r"^The Ramayana/Book ([IVX]+)/Canto ([IVXLC]+)[\.:]?\s*(.*)$"),
                     book=lambda m: roman(m.group(1)), chapter=lambda m: roman(m.group(2)),
                     translator="R. T. H. Griffith"),
    "mahabharata": dict(prefix="mbh", page_prefix="The Mahabharata/",
                        title=re.compile(r"^The Mahabharata/Book (\d+): ([^/]+)/Section (\d+)$"),
                        book=lambda m: int(m.group(1)), chapter=lambda m: int(m.group(3)),
                        translator="Kisari Mohan Ganguli"),
}


def api(params, tries=5):
    params = dict(params, format="json")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=UA)
    for k in range(tries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def list_pages(prefix):
    pages, cont = [], {}
    while True:
        d = api(dict(action="query", list="allpages", apprefix=prefix, aplimit=500, apnamespace=0, **cont))
        pages += [p["title"] for p in d["query"]["allpages"]]
        if "continue" not in d:
            return pages
        cont = {"apcontinue": d["continue"]["apcontinue"]}


def clean(wikitext):
    t = re.sub(r"\{\{[Hh]eader.*?^\s*\}\}", "", wikitext, flags=re.S | re.M)
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"\[\[[a-z\-]+:[^\]]*\]\]", "", t)                 # interwiki links ([[sa:...]])
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"^\s*[=*#|].*$", "", t, flags=re.M)      # headings, lists, tables
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def fetch(key):
    w = WORKS[key]
    titles = [t for t in list_pages(w["page_prefix"]) if w["title"].match(t)]
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f"{key}.jsonl")
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line); done[r["source_title"]] = r
    print(f"{key}: {len(titles)} chapter pages, {len(done)} cached", flush=True)
    rows = list(done.values())
    for t in titles:
        if t in done:
            continue
        m = w["title"].match(t)
        d = api(dict(action="query", prop="revisions", rvprop="content", rvslots="main", titles=t))
        page = list(d["query"]["pages"].values())[0]
        txt = ((page.get("revisions") or [{}])[0].get("slots", {}).get("main", {}).get("*", ""))
        book, ch = w["book"](m), w["chapter"](m)
        rows.append(dict(adhyaya_id=f"{w['prefix']}-{book}-{ch}", book_no=book, adhyaya_no=ch,
                         title_en=(m.group(3).strip() if key == "ramayana" else m.group(2).strip()),
                         text_en=clean(txt), translator=w["translator"], source_title=t,
                         source_url="https://en.wikisource.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")),
                         license=LICENSE))
        time.sleep(0.2)
    rows.sort(key=lambda r: (r["book_no"], r["adhyaya_no"]))
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(f"{key}: {len(rows)} chapters written; per book {dict(collections.Counter(r['book_no'] for r in rows))}; "
          f"chars {sum(len(r['text_en']) for r in rows)}")


if __name__ == "__main__":
    for k in sys.argv[1:] or ["ramayana"]:
        fetch(k)
