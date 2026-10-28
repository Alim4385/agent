"""HTML → əsas mətn çıxarıcısı (trafilatura əvəzi, standart kitabxana ilə).

Məqsəd: səhifədən naviqasiya, reklam, footer kimi "zibil" hissələri atıb
məqalənin əsl məzmununu çıxarmaq. Alqoritm sadə və proqnozlaşdırılandır:

  1) `html.parser` ilə sənəd gəzilir, mətn "blok"lara bölünür
     (p, div, li, h1-h6, td, blockquote ...).
  2) script/style/nav/aside/footer/form kimi konteynerlər tamamilə atılır.
  3) Hər blok üçün "link sıxlığı" hesablanır — menyu və keçid siyahıları
     mətnin çoxunu linklərdə saxladığı üçün avtomatik kənarlaşdırılır.
  4) `<article>`/`<main>` daxilində kifayət qədər mətn varsa, yalnız onlar
     götürülür; yoxsa bütün "keçərli" bloklar sırayla birləşdirilir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List, Optional, Tuple

# Mətn bloklarını ayıran teqlər
BLOCK_TAGS = frozenset(
    "p div li ul ol dl dd dt h1 h2 h3 h4 h5 h6 tr td th table section article "
    "main blockquote pre br hr figure figcaption address details summary "
    "header footer nav aside form fieldset legend caption".split()
)

# Bu konteynerlərin içi tamamilə atılır (mətn deyil, kod/UI)
HARD_SKIP = frozenset(
    "script style noscript template svg canvas iframe object embed "
    "nav aside footer form button select input textarea option dialog menu".split()
)

# Yumşaq: səhifə header-i adətən logo/menyu olur, amma bəzən başlıq da olur —
# yalnız başqa mətn yoxdursa istifadə olunur.
SOFT_SKIP = frozenset({"header"})

# HTML-də bağlanmayan (void) teqlər — bunlar üçün "skip" sayğacı artırılmamalıdır,
# əks halda sayğac heç vaxt sıfırlanmır və qalan bütün səhifə atılır.
VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


@dataclass
class Block:
    text: str
    link_chars: int
    in_article: bool
    soft: bool
    tag: str

    @property
    def link_ratio(self) -> float:
        return self.link_chars / max(1, len(self.text))


class _BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: List[Block] = []
        self.title_parts: List[str] = []
        self._in_title = False
        self._skip = 0
        self._soft = 0
        self._article = 0
        self._links = 0
        self._buf: List[str] = []
        self._buf_link_chars = 0
        self._buf_article = False
        self._buf_soft = False
        self._buf_tag = ""

    # -- köməkçilər ------------------------------------------------------
    def _flush(self) -> None:
        text = _normalize("".join(self._buf))
        block = Block(
            text=text,
            link_chars=self._buf_link_chars,
            in_article=self._buf_article,
            soft=self._buf_soft,
            tag=self._buf_tag,
        )
        self._buf, self._buf_link_chars = [], 0
        self._buf_article, self._buf_soft, self._buf_tag = False, False, ""
        if text:
            self.blocks.append(block)

    def _ensure_open(self) -> None:
        """Blokun "kontekstini" ilk mətn parçasında qeyd edirik."""
        if not self._buf:
            self._buf_article = self._article > 0
            self._buf_soft = self._soft > 0

    def _open_block(self, tag: str) -> None:
        self._ensure_open()
        if not self._buf_tag:  # <h1> kimi teq məlumatı mətn tərəfindən silinməsin
            self._buf_tag = tag

    # -- HTMLParser API --------------------------------------------------
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag in HARD_SKIP:
            if tag not in VOID_TAGS:
                self._skip += 1
            return
        if tag in SOFT_SKIP:
            self._soft += 1
        if tag in ("article", "main"):
            self._article += 1
        if tag == "a":
            self._links += 1
        if tag in BLOCK_TAGS:
            self._flush()
            if tag != "br":
                self._open_block(tag)
            else:
                self._buf_tag = "br"

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        # <br/>, <img/>, <input/> kimi özünü bağlayan teqlər: sayğac dəyişmir
        if tag in BLOCK_TAGS and tag not in HARD_SKIP:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in HARD_SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag in SOFT_SKIP:
            self._soft = max(0, self._soft - 1)
        if tag in ("article", "main"):
            self._article = max(0, self._article - 1)
        if tag == "a":
            self._links = max(0, self._links - 1)
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip:
            return
        if not data or not data.strip():
            if self._buf and data:
                self._buf.append(data)  # sözlər arasındakı boşluğu saxlayırıq
            return
        self._ensure_open()
        self._buf.append(data)
        if self._links:
            self._buf_link_chars += len(data)


def _is_content(block: Block, min_len: int = 30) -> bool:
    """Blok məzmun sayılırmı: kifayət qədər uzun və əsasən link olmayan."""
    if len(block.text) < min_len:
        return False
    return block.link_ratio < 0.5


def extract_text(html: str, *, min_article_chars: int = 200) -> str:
    """HTML sənədindən əsas mətni çıxarır. Uyğun heç nə tapılmasa boş sətir qaytarır."""
    if not html or not html.strip():
        return ""

    parser = _BlockParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — pozulmuş HTML bizi dayandırmamalıdır
        pass
    parser._flush()  # noqa: SLF001 — axırıncı bloku da yazırıq

    blocks = [b for b in parser.blocks if b.text]
    if not blocks:
        return ""

    def usable(block: Block) -> bool:
        if _is_content(block):
            return True
        # Qısa başlıqlar (h1-h6) faydalıdır — link deyilsə saxlayırıq
        return block.tag in ("h1", "h2", "h3", "h4", "h5", "h6") and len(block.text) >= 3 and block.link_ratio < 0.5

    article_blocks = [b for b in blocks if b.in_article and not b.soft and usable(b)]
    if sum(len(b.text) for b in article_blocks) >= min_article_chars:
        chosen = article_blocks
    else:
        chosen = [b for b in blocks if not b.soft and usable(b)]
        if not chosen:  # çox kiçik səhifə — header də daxil olmaqla hər şey
            chosen = [b for b in blocks if usable(b)] or blocks

    lines: List[str] = []
    seen = set()
    for block in chosen:
        key = block.text.lower()
        if key in seen:  # təkrarlanan bloklar (mobil/desktop variantları) atılır
            continue
        seen.add(key)
        lines.append(block.text)

    return "\n\n".join(lines).strip()


class _TextOnlyParser(HTMLParser):
    """Ehtiyat rejim: yalnız script/style atılır, qalan bütün mətn yığılır."""

    _SKIP = frozenset({"script", "style", "noscript", "template", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _plain_text(html: str) -> str:
    parser = _TextOnlyParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001
        pass
    return "\n\n".join(parser.parts[:400]).strip()


def extract_title(html: str) -> str:
    """<title> mətnini qaytarır (meta paylaşım üçün faydalıdır)."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    return _normalize(match.group(1)) if match else ""
