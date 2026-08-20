"""Rule-based sentence tokenizer for Indic-script (and Latin/Arabic-script) text.

IndicCorpV2 documents are paragraphs, not sentences. We split on the sentence-final
punctuation used across the scripts present in the corpus:
  - Devanagari/Bengali/etc danda:  । ॥
  - Latin:                          . ! ?
  - Arabic (Urdu/Kashmiri/Sindhi):  ۔ ؟
  - Misc:                           ‽
"""
import re

_SENT_END = re.compile(r'([।॥.!?؟۔‽]+)')
_WS = re.compile(r'\s+')


def split_paragraph(text: str, min_chars: int = 8, min_words: int = 3, max_words: int = 80):
    if not text:
        return []
    text = _WS.sub(' ', text).strip()
    if not text:
        return []

    parts = _SENT_END.split(text)
    sentences = []
    buf = ''
    for part in parts:
        if _SENT_END.fullmatch(part):
            buf = (buf + part).strip()
            if buf:
                sentences.append(buf)
            buf = ''
        else:
            buf += part
    if buf.strip():
        sentences.append(buf.strip())

    out = []
    for s in sentences:
        s = s.strip()
        if len(s) < min_chars:
            continue
        n_words = len(s.split(' '))
        if n_words < min_words or n_words > max_words:
            continue
        out.append(s)
    return out
