"""Neuronte for pictures: still images and PDF pages drawn with Neuronte's nets.

Two outputs, both drawn only by the nets (no font file is read):

cells   one character, a foreground and a background colour per terminal
        cell, for any true-colour terminal. Every candidate character's
        coverage comes from the same bundles the terminal draws with, at the
        terminal's own cell size, so the fit is judged on exactly what will
        appear. Ranking is exact (TwoInk.rank), then the shortlist is fitted
        with box-limited least squares (TwoInk.fit).
pixels  MultiInk at 2 x 4 pixel cells, drawn by the three neural masks of
        neuronte-video-masks-v0 and sent with the kitty graphics protocol.

Sources: anything Pillow opens (JPEG, PNG, WebP, GIF first frame, ...) and
PDF pages through poppler.
"""
from pathlib import Path
import base64
import hashlib
import os
import sys
import zlib
import numpy as np

HERE = Path(__file__).resolve().parent


def _neuronte_home():
    """The neuronte-v2 tree to draw with: NEURONTE_HOME, else where Kilix
    installs it. Its library must be built (build/libneuronte.so)."""
    for cand in (os.environ.get('NEURONTE_HOME'), Path.home() / '.local/gpu_terminal/sources/neuronte'):
        if cand and (Path(cand) / 'tools/neuronte_video.py').exists():
            return Path(cand).resolve()
    raise SystemExit('neuronte-view: set NEURONTE_HOME to a built neuronte-v2 tree')


ROOT = _neuronte_home()
sys.path.insert(0, str(ROOT / 'tools'))
import neuronte_video as nv  # noqa: E402
import glyphs  # noqa: E402  (neuronte-v2's train/, put on the path by neuronte_video)

WEIGHTS = ROOT / 'weights'
CACHE = Path(os.environ.get('NEURONTE_VIEW_CACHE', Path.home() / '.cache/neuronte-view'))


# ---------------------------------------------------------------- candidates

def _ranges(spec):
    out = []
    for part in spec.replace(',', ' ').split():
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition('..')
        lo = int(lo.removeprefix('U+'), 16)
        hi = int(hi.removeprefix('U+'), 16) if hi else lo
        out.extend(range(lo, hi + 1))
    return out


def bundle_members(name):
    """Scalars a bundle was trained on, from its .list.txt."""
    cps = []
    for line in (WEIGHTS / f'{name}.list.txt').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            cps.extend(_ranges(line))
    return cps


def _pixel_set():
    cps = [0x20] + [ord(ch) for ch, _, _ in glyphs.charset('pixel')]
    return sorted(set(cps))


def _draw_set():
    cps = set(_pixel_set())
    cps.update(ord(ch) for ch, _, _ in glyphs.charset('draw'))
    # Shapes useful for pictures, wherever a term bundle draws them:
    # geometric triangles and quarter circles, the legacy-computing wedges
    # and diagonals, and the Powerline separators.
    for lo, hi in ((0x25E2, 0x25E5), (0x25F8, 0x25FA), (0x25FF, 0x25FF), (0x25D6, 0x25D7),
                   (0x25E0, 0x25E1), (0x1FB3C, 0x1FBAF), (0xE0B0, 0xE0BF)):
        cps.update(range(lo, hi + 1))
    return sorted(cps)


SETS = {
    'pixel': _pixel_set,   # space, blocks, quadrants, sextants, octants
    'draw': _draw_set,     # + box, shades, braille, triangles, wedges where drawn
    'full': None,          # every one-cell scalar the bundles draw
}


def bundle_order():
    """As neuroterm: the newest split file, then every term bundle by name."""
    splits = sorted(WEIGHTS.glob('neuronte-split-v*.nrt'), key=lambda p: int(p.stem.rsplit('v', 1)[1]))
    terms = sorted(p.stem for p in WEIGHTS.glob('neuronte-term-*.nrt'))
    return [splits[-1].stem] + terms


class Bundles:
    """Coverage for any scalar from the bundle the terminal would draw it with:
    the split file first, then the term bundles, as neuroterm routes it."""
    ORDER = bundle_order()

    def __init__(self, library=None):
        self.nets = {}
        for name in self.ORDER:
            path = WEIGHTS / f'{name}.nrt'
            self.nets[name] = nv.NeuralMasks(path, library)
        lib = self.nets[self.ORDER[0]].lib
        self.untrained = getattr(lib, 'neuronte_untrained', None)
        if self.untrained is not None:
            import ctypes as C
            self.untrained.argtypes = [C.c_void_p, C.c_uint32]
            self.untrained.restype = C.c_int
            # neuroterm sets this on the split file, so characters it was not
            # trained on go on to the term bundles instead of its mark.
            setu = lib.neuronte_set_untrained
            setu.argtypes = [C.c_void_p, C.c_int]
            setu(self.nets[self.ORDER[0]].handle, 1)

    def trained(self, name, cp):
        if self.untrained is None:
            return True
        return not self.untrained(self.nets[name].handle, cp)

    def who(self, cp):
        for name in self.ORDER:
            if self.trained(name, cp):
                return name
        return None

    def mask(self, cp, w, h):
        name = self.who(cp)
        if name is None:
            return None
        return self.nets[name].mask(cp, w, h)

    def all_scalars(self):
        cps = set()
        data = (WEIGHTS / f'{self.ORDER[0]}.nrt').read_bytes()
        import struct
        nets, ranges = struct.unpack_from('<II', data, 12)
        off = 28 + 4 * nets
        for i in range(ranges):
            lo, hi, _ = struct.unpack_from('<III', data, off + 12 * i)
            cps.update(range(lo, min(hi, 0x10FFFF) + 1))
        for name in self.ORDER[1:]:
            cps.update(bundle_members(name))
        return sorted(cp for cp in cps if not 0xD800 <= cp <= 0xDFFF and cp >= 0x20
                      and not 0x7F <= cp < 0xA0 and _one_cell(cp))

    def close(self):
        for n in self.nets.values():
            n.close()


def _one_cell(cp):
    import unicodedata
    ch = chr(cp)
    if unicodedata.category(ch) in ('Mn', 'Me', 'Cf', 'Cc', 'Zl', 'Zp', 'Cs', 'Cn'):
        return False
    return unicodedata.east_asian_width(ch) not in ('W', 'F')


def candidate_masks(set_name, cell_w, cell_h, library=None):
    """(scalars, masks K x P float32), cached on disk by weights hash and size."""
    h = hashlib.sha256()
    for name in Bundles.ORDER:
        h.update((WEIGHTS / f'{name}.nrt').read_bytes())
    h.update(f'{set_name}:{cell_w}x{cell_h}:v1'.encode())
    path = CACHE / f'{set_name}-{cell_w}x{cell_h}-{h.hexdigest()[:16]}.npz'
    if path.exists():
        z = np.load(path)
        return z['scalars'], z['masks']
    b = Bundles(library)
    try:
        cps = b.all_scalars() if set_name == 'full' else SETS[set_name]()
        keep, masks = [], []
        for cp in cps:
            m = b.mask(cp, cell_w, cell_h)
            if m is None:
                continue
            keep.append(cp)
            masks.append(m.ravel())
    finally:
        b.close()
    scalars = np.array(keep, np.uint32)
    masks = np.stack(masks).astype(np.float32)
    # Identical masks add nothing; keep the first (lowest scalar).
    _, first = np.unique(np.round(masks, 3), axis=0, return_index=True)
    first.sort()
    scalars, masks = scalars[first], masks[first]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scalars=scalars, masks=masks)
    return scalars, masks


def _rank(self, rgb, k=16, always=None):
    """The same exact shortlist as TwoInk.rank in neuronte-v2 after 2.5.0,
    for trees that predate it."""
    if getattr(self, '_unit', None) is None:
        m = self.masks - self.masks.mean(1, keepdims=True)
        norm = np.sqrt((m * m).sum(1, keepdims=True))
        self._unit = np.divide(m, norm, out=np.zeros_like(m), where=norm > 1e-9).astype(np.float32)
    p = nv.patches(rgb, self.width, self.height)
    grid = p.shape[:2]
    x = p.reshape(-1, self.width * self.height, 3).astype(np.float32)
    x -= x.mean(1, keepdims=True)
    k = max(1, min(k, len(self.masks)))
    out = np.empty((len(x), k), np.int64)
    for start in range(0, len(x), 512):
        s = np.matmul(self._unit[None], x[start:start + 512])
        score = (s * s).sum(2)
        out[start:start + len(score)] = (np.argpartition(-score, k - 1, axis=1)[:, :k]
                                         if k < score.shape[1] else np.argsort(-score, 1))
    if always is not None:
        always = np.asarray(always, np.int64).ravel()
        out = np.concatenate([out, np.broadcast_to(always, (len(out), len(always)))], 1)
    return out.reshape(*grid, -1)


if not hasattr(nv.TwoInk, 'rank'):
    nv.TwoInk.rank = _rank


class Cells(nv.TwoInk):
    """TwoInk over a named candidate set, with exact ranking."""

    def __init__(self, set_name='draw', cell_w=10, cell_h=20, k=12, library=None):
        self.width, self.height, self.margin = cell_w, cell_h, 0
        self.scalars, masks = candidate_masks(set_name, cell_w, cell_h, library)
        self.masks = masks.astype(np.float64)
        self.previous = None
        self._unit = None
        self.k = k
        flat = np.where(masks.std(1) < 1e-6)[0]
        self.flat = flat[:1] if len(flat) else np.array([], np.int64)

    def fit(self, rgb, proposals=None):
        if proposals is None and self.k:
            proposals = self.rank(rgb, self.k, always=self.flat)
        self.previous = None
        return super().fit(rgb, proposals)


# ------------------------------------------------------------------ sources

def open_image(path):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (0, 0, 0, 255))
        bg.alpha_composite(im)
        im = bg
    return np.asarray(im.convert('RGB'))


def pdf_pages(path):
    import gi
    gi.require_version('Poppler', '0.18')
    from gi.repository import Poppler
    doc = Poppler.Document.new_from_file(Path(path).resolve().as_uri(), None)
    return doc


def open_pdf_page(path, page=0, width_px=1600, doc=None):
    """Rasterise one PDF page with poppler at width_px, on white.

    Note: poppler draws the page's own text with the fonts embedded in the
    PDF, so this raster is not font-free."""
    import cairo
    doc = doc or pdf_pages(path)
    p = doc.get_page(page)
    w, h = p.get_size()
    s = width_px / w
    W, H = int(round(w * s)), int(round(h * s))
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.scale(s, s)
    p.render(ctx)
    surf.flush()
    a = np.ndarray((H, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :W]
    return a[:, :, [2, 1, 0]].copy()


def fit_size(shape, cols, rows, cell_w, cell_h):
    """Grid that fits the picture in cols x rows cells, keeping its aspect."""
    h, w = shape[:2]
    c = cols
    r = max(1, round(c * (h / w) * cell_w / cell_h))
    if rows and r > rows:
        r = rows
        c = max(1, round(r * (w / h) * cell_h / cell_w))
    return c, r


def resample(rgb, width, height, linear=True):
    """Lanczos resampling; in linear light by default, so detail averages
    the way light does instead of darkening."""
    from PIL import Image
    if not linear:
        return np.asarray(Image.fromarray(rgb).resize((width, height), Image.LANCZOS))
    lin = _to_linear(rgb)
    chans = [np.asarray(Image.fromarray(lin[:, :, c], 'F').resize((width, height), Image.LANCZOS))
             for c in range(3)]
    return _to_srgb(np.stack(chans, -1))


def _to_linear(rgb):
    x = rgb.astype(np.float32) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _to_srgb(lin):
    x = np.clip(lin, 0, 1)
    y = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ outputs

def ansi_rows(cells, scalars):
    """True-colour SGR text, one line per row, relative (no cursor moves),
    so it can be printed into scrollback like icat/chafa/tiv output."""
    idx, col = cells['indices'], cells['colours']
    lines = []
    for y in range(idx.shape[0]):
        out, last = [], None
        for x in range(idx.shape[1]):
            bg, fg = col[y, x]
            key = (tuple(fg), tuple(bg))
            if key != last:
                out.append('\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm' % (*fg, *bg))
                last = key
            out.append(chr(int(scalars[idx[y, x]])))
        out.append('\x1b[0m')
        lines.append(''.join(out))
    return '\n'.join(lines) + '\n'


def kitty_image(rgb, cols, rows, image_id=None):
    """Kitty graphics protocol, inline zlib RGB, placed over cols x rows."""
    h, w = rgb.shape[:2]
    data = base64.standard_b64encode(zlib.compress(np.ascontiguousarray(rgb).tobytes(), 6))
    out = []
    first = True
    for i in range(0, len(data), 4096):
        chunk = data[i:i + 4096]
        more = 1 if i + 4096 < len(data) else 0
        if first:
            ctl = f'a=T,f=24,o=z,s={w},v={h},c={cols},r={rows},q=2,m={more}'
            if image_id:
                ctl += f',i={image_id}'
            first = False
        else:
            ctl = f'm={more}'
        out.append(f'\x1b_G{ctl};{chunk.decode()}\x1b\\')
    return ''.join(out) + '\n'


def pixels_multiink(rgb, library=None):
    """Draw the picture through MultiInk's neural masks at 2 x 4 cells."""
    mi = nv.MultiInk(library=library)
    h, w = rgb.shape[:2]
    H, W = h - h % mi.height, w - w % mi.width
    rgb = np.ascontiguousarray(rgb[:H, :W])
    out = np.empty_like(rgb)
    mi.render_rgb24(rgb, out)
    return out


def png(path, rgb):
    from PIL import Image
    Image.fromarray(rgb).save(path)


# ------------------------------------------------- PDF pages without fonts

class Faces:
    """Neuronte's proportional desktop faces, chosen by em as neuronte-fontfile
    does: a hand-drawn face per em up to 16 px, the scaled face to 32, the
    large face above. Each face's cap height and baseline are measured from
    its own drawing of 'H', never assumed."""

    def __init__(self, library=None):
        import ctypes as C
        self.C = C
        self.lib = C.CDLL(str(library or ROOT / 'build/libneuronte.so'))
        L = self.lib
        L.neuronte_open_file.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        L.neuronte_draw_text.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                         C.c_int, C.c_int, C.c_char_p, C.c_size_t, C.c_uint32]
        L.neuronte_text_width.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_int, C.c_int]
        self.handles = {}
        self.metrics = {}

    def _newest(self, pattern):
        found = sorted(WEIGHTS.glob(pattern))
        return found[-1] if found else None

    def face(self, em):
        if em <= 16:
            path = self._newest(f'neuronte-ui{em}-v*.nrt')
        elif em <= 32:
            path = self._newest('neuronte-ui-scaled-latin1-v*.nrt')
        else:
            path = self._newest('neuronte-ui-large-latin1-v*.nrt')
        if path is None:
            raise FileNotFoundError(f'no Neuronte face for em {em}')
        if path not in self.handles:
            h = self.C.c_void_p()
            if self.lib.neuronte_open_file(str(path).encode(), self.C.byref(h)):
                raise ValueError(f'cannot open {path}')
            self.handles[path] = h
        return self.handles[path]

    def measure(self, em):
        """(cap height, baseline) in pixels below the pen, from drawing 'H'."""
        if em not in self.metrics:
            buf = np.zeros((2 * em + 4, 2 * em + 4), np.uint32)
            self.lib.neuronte_draw_text(self.face(em), buf.ctypes.data, buf.shape[1], buf.shape[1], buf.shape[0],
                                        0, 0, em, 0, b'H', 1, 0xFF000000)
            ink = np.where((buf >> 24) > 127)[0]
            self.metrics[em] = (int(ink[-1] - ink[0] + 1), int(ink[-1] + 1)) if len(ink) else (em, int(1.5 * em))
        return self.metrics[em]

    def em_for_cap(self, cap_px):
        """The em whose drawn capital is closest to cap_px tall."""
        guess = max(1, round(cap_px / 0.7))
        best = None
        for em in range(max(1, guess - 6), guess + 7):
            try:
                d = abs(self.measure(em)[0] - cap_px)
            except FileNotFoundError:
                continue
            if best is None or d < best[0]:
                best = (d, em)
        return best[1] if best else guess

    def draw(self, argb, x, baseline, cap_px, text, rgb, bold=0, width=None):
        """Draw text with its baseline at `baseline`, blended over argb (H x W
        uint32). With `width`, the text is centred in that many pixels from x,
        so a character keeps the place the page gave it."""
        em = self.em_for_cap(cap_px)
        _, base = self.measure(em)
        data = text.encode()
        h, w = argb.shape
        colour = 0xFF000000 | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
        if width is not None:
            x += (width - self.lib.neuronte_text_width(self.face(em), data, len(data), bold, em)) / 2
        return self.lib.neuronte_draw_text(self.face(em), argb.ctypes.data, w, w, h,
                                           int(round(x)), int(round(baseline - base)), em, bold, data, len(data), colour)


def open_pdf_page_neuronte(path, page=0, width_px=1600, faces=None):
    """A PDF page with no font file used: the page's own text is left out of
    the raster (its glyphs are removed from poppler's SVG of the page) and
    drawn again, character by character at the page's own places, baselines,
    colours and weights, by
    Neuronte's proportional faces. Pictures and vector drawing stay.

    Returns (rgb, chars) where chars lists (character, x, baseline, cap) in pixels."""
    import re
    import subprocess
    import tempfile
    import cairo
    import gi
    gi.require_version('Poppler', '0.18')
    gi.require_version('Rsvg', '2.0')
    from gi.repository import Poppler, Rsvg

    doc = pdf_pages(path)
    p = doc.get_page(page)
    pw, ph = p.get_size()
    s = width_px / pw
    W, H = int(round(pw * s)), int(round(ph * s))

    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / 'page.svg'
        subprocess.run(['pdftocairo', '-svg', '-f', str(page + 1), '-l', str(page + 1), str(path), str(svg)], check=True)
        text = svg.read_text()
    uses = [(float(x), float(y)) for x, y in
            re.findall(r'<use xlink:href="#glyph-[^"]*" x="([-0-9.]+)" y="([-0-9.]+)"', text)]
    stripped = re.sub(r'<use xlink:href="#glyph-[^"]*"[^>]*/>', '', text)
    handle = Rsvg.Handle.new_from_data(stripped.encode())
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    vp = Rsvg.Rectangle()
    vp.x, vp.y, vp.width, vp.height = 0, 0, W, H
    handle.render_document(ctx, vp)
    surf.flush()
    argb = np.ndarray((H, surf.get_stride() // 4), np.uint32, surf.get_data())[:, :W].copy()

    # Every character at the place the page gives it: its box from poppler's
    # text layout, its baseline from the glyph poppler drew there, its size,
    # colour and weight from the text attributes.
    chars = p.get_text()
    ok, rects = p.get_text_layout()
    attrs = p.get_text_attributes()
    n = min(len(chars), len(rects))
    colour_at = np.zeros(n, np.uint32)
    size_at = np.zeros(n, np.float64)
    bold_at = np.zeros(n, np.int32)
    for a in attrs:
        c = a.color
        sl = slice(a.start_index, min(a.end_index + 1, n))
        colour_at[sl] = ((c.red >> 8) << 16) | ((c.green >> 8) << 8) | (c.blue >> 8)
        size_at[sl] = a.font_size
        bold_at[sl] = 1 if re.search(r'bold|black|heavy|semibold', a.font_name or '', re.I) else 0
    uses_a = np.array(uses) if uses else np.zeros((0, 2))
    faces = faces or Faces()
    words = []
    for i in range(n):
        ch = chars[i]
        if ch.isspace():
            continue
        r = rects[i]
        size = size_at[i] or (r.y2 - r.y1)
        if len(uses_a):
            near = uses_a[(np.abs(uses_a[:, 0] - r.x1) < 0.3 * size + 0.5) & (uses_a[:, 1] >= r.y1) & (uses_a[:, 1] <= r.y2 + 0.5)]
        else:
            near = uses_a
        base = float(near[np.argmin(np.abs(near[:, 0] - r.x1)), 1]) if len(near) else r.y2 - 0.22 * (r.y2 - r.y1)
        c = int(colour_at[i])
        cap = 0.7 * size * s
        faces.draw(argb, r.x1 * s, base * s, cap, ch, ((c >> 16) & 255, (c >> 8) & 255, c & 255),
                   bold=int(bold_at[i]), width=(r.x2 - r.x1) * s)
        words.append((ch, r.x1 * s, base * s, cap))
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[..., 0] = (argb >> 16) & 255
    rgb[..., 1] = (argb >> 8) & 255
    rgb[..., 2] = argb & 255
    return rgb, words
