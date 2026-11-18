#!/usr/bin/env python3
"""Checks for tools/neuronte_view.py and TwoInk.rank.

1. rank() is exact: its first choice for every cell is the brute-force
   argmin of the unconstrained two-colour error over all candidates.
2. A shortlist of every candidate fits with the same error and pixels as none.
3. Cells' characters all come from a bundle the terminal draws (no mark).
4. A PDF page drawn without fonts holds no glyph from the PDF's fonts and
   still has text (Neuronte's) where the page has text.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neuronte_view as V  # noqa: E402

failures = 0


def check(ok, what):
    global failures
    print(('ok   ' if ok else 'FAIL ') + what)
    failures += not ok


rng = np.random.default_rng(7)
cells = V.Cells('pixel', 6, 12, k=8)
img = (rng.random((12 * 5, 6 * 7, 3)) * 255).astype(np.uint8)
img[:24] = np.linspace(0, 255, 6 * 7).astype(np.uint8)[None, :, None]  # a smooth ramp too

# 1. exactness of rank against brute force
p = V.nv.patches(img, 6, 12).reshape(-1, 72, 3).astype(np.float64)
m = cells.masks
X = np.concatenate([np.ones((len(m), 72, 1)), m[:, :, None]], 2)       # K x P x 2
err = np.empty((len(p), len(m)))
for k in range(len(m)):
    A = X[k]
    coef, *_ = np.linalg.lstsq(A, p.transpose(1, 0, 2).reshape(72, -1), rcond=None)
    r = p.transpose(1, 0, 2).reshape(72, -1) - A @ coef
    err[:, k] = (r * r).reshape(72, len(p), 3).sum((0, 2))
best = err.min(1)
top = cells.rank(img, 1).reshape(-1)
check(np.allclose(err[np.arange(len(p)), top], best, rtol=1e-4, atol=1e-3),
      'rank top-1 equals the brute-force unconstrained optimum in every cell')

# 2. full shortlist == no shortlist
everything = np.broadcast_to(np.arange(len(m)), (5, 7, len(m)))
a = V.nv.TwoInk.fit(cells, img, None)
b = V.nv.TwoInk.fit(cells, img, everything)
# Candidates can tie, so compare each cell's error and the drawn pixels.
check(np.allclose(a['sse'], b['sse'], rtol=1e-9, atol=1e-6) and np.array_equal(cells.decode(a), cells.decode(b)),
      'a shortlist of every candidate fits as well as none, and draws the same pixels')

# 3. every chosen character is drawn by a bundle
d = V.Cells('draw', 10, 20)
out = d.fit(V.resample(img, 10 * 8, 20 * 3))
b2 = V.Bundles()
chosen = {int(d.scalars[i]) for i in np.unique(out['indices'])}
check(all(b2.who(cp) is not None for cp in chosen), f'{len(chosen)} chosen characters all drawn by a bundle')
check(all(b2.who(int(cp)) is not None for cp in d.scalars), f'all {len(d.scalars)} draw-set candidates drawn by a bundle')

# 4. font-free PDF page
try:
    import cairo
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / 't.pdf'
        surf = cairo.PDFSurface(str(pdf), 200, 100)
        ctx = cairo.Context(surf)
        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle(10, 60, 50, 30)
        ctx.fill()
        ctx.move_to(10, 40)
        ctx.set_font_size(20)
        ctx.show_text('Hello')
        surf.finish()
        rgb, chars = V.open_pdf_page_neuronte(pdf, 0, 400)
        check(''.join(c for c, *_ in chars) == 'Hello', 'the page text is found: ' + ''.join(c for c, *_ in chars))
        ink = (rgb.min(2) < 128)
        check(ink[120:180, 20:120].mean() > 0.9, 'the drawn rectangle stays')
        check(ink[40:85, 15:120].any(), 'Neuronte drew text where the page has text')
except ImportError as e:
    print('skip PDF check:', e)

sys.exit(1 if failures else 0)
