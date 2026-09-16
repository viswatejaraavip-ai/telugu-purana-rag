# Corpus: 17 Puranas in Sanskrit (IAST, Telugu script, Devanagari)

One record per verse in `<work>.jsonl`, one record per adhyaya in `<work>_adhyayas.jsonl`,
provenance and checksums in `sources.json`, on-demand Telugu meanings in `tatparyam_cache.jsonl`
(status `unreviewed`). Built by `scripts/build_gretil_purana.py`.

| work | adhyayas | verses |
|---|---|---|
| bhagavata | 335 | 13,986 |
| vishnu | 126 | 6,383 |
| markandeya | 90 | 4,522 |
| garuda | 317 | 11,957 |
| narasimha | 68 | 3,435 |
| shiva (samhitas 1, 7) | 101 | 5,664 |
| kurma | 95 | 5,821 |
| vamana + saromahatmya | 69 + 26 | 5,683 |
| matsya (1–176) | 176 | 8,515 |
| linga (purva) | 108 | 6,831 |
| agni | 381 | 11,030 |
| brahma | 246 | 13,258 |
| narada | 167 | 15,580 |
| brahmanda | 156 | 13,709 |
| skanda reva khanda | 116 | 6,724 |
| vayu reva khanda | 232 | 7,775 |

## Source and licence

The Sanskrit text is from the **Göttingen Register of Electronic Texts in Indian Languages (GRETIL)**,
https://gretil.sub.uni-goettingen.de/ — see `sources.json` for the exact file, version and SHA-256 of
each work. GRETIL distributes these e-texts under **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0)**. The files in this directory are derived from them (parsed into
verse records, transliterated from IAST into Telugu script and Devanagari) and are therefore distributed
under the same licence: attribution required, **non-commercial use only**, share-alike.

The underlying Sanskrit works are in the public domain; the licence attaches to GRETIL's digitisation.
For commercial use, re-source the same texts from a public-domain digitisation (e.g. Ambuda or
Sanskrit Wikisource) and re-run the parser; verse ids follow the editions and will match.

The code in this repository (outside `corpus/`) is MIT-licensed; see `../LICENSE`.
