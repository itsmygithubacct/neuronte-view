#!/usr/bin/env python3
"""neuronte-view: show a picture or a PDF page in the terminal, drawn by Neuronte's nets.

  neuronte-view.py photo.jpg                  # cells, fitted to the terminal width
  neuronte-view.py --cols 100 photo.jpg
  neuronte-view.py --mode pixels photo.jpg    # kitty graphics (Kilix), MultiInk masks
  neuronte-view.py --page 3 paper.pdf
  neuronte-view.py --preview out.png photo.jpg  # also write what the cells look like

Cells mode needs a true-colour terminal that draws with Neuronte (Kilix with
KILIX_NEURONTE_ONLY, neuroterm) to look exactly as fitted; in another
terminal the same characters are drawn by its font.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuronte_view as V  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file')
    ap.add_argument('--mode', choices=('cells', 'pixels'), default='cells')
    ap.add_argument('--set', choices=tuple(V.SETS), default='draw',
                    help='candidate characters for cells mode (default draw)')
    ap.add_argument('--cols', type=int, default=0, help='width in cells (default: terminal width)')
    ap.add_argument('--rows', type=int, default=0, help='height limit in cells (default: none)')
    ap.add_argument('--cell', default='10x20', help="terminal cell size in pixels, WxH (neuroterm's default 10x20)")
    ap.add_argument('--k', type=int, default=12, help='exact shortlist length before the bounded fit')
    ap.add_argument('--page', type=int, default=1, help='PDF page, from 1')
    ap.add_argument('--srgb-resample', action='store_true', help='resample in sRGB, not linear light')
    ap.add_argument('--preview', help='write a PNG of the result as drawn by the nets')
    ap.add_argument('--stats', action='store_true', help='timings and error on stderr')
    a = ap.parse_args(argv)

    cw, ch = (int(v) for v in a.cell.lower().split('x'))
    cols = a.cols or shutil.get_terminal_size((80, 24)).columns
    t0 = time.perf_counter()
    path = Path(a.file)
    if path.suffix.lower() == '.pdf':
        src = V.open_pdf_page(path, a.page - 1, width_px=max(1600, cols * cw * 2))
    else:
        src = V.open_image(path)
    t1 = time.perf_counter()

    if a.mode == 'pixels':
        c, r = V.fit_size(src.shape, cols, a.rows, cw, ch)
        img = V.resample(src, c * cw, r * ch, linear=not a.srgb_resample)
        drawn = V.pixels_multiink(img)
        sys.stdout.write(V.kitty_image(drawn, c, r))
        if a.preview:
            V.png(a.preview, drawn)
        t2 = time.perf_counter()
        if a.stats:
            print(f'decode {1e3*(t1-t0):.0f} ms, draw {1e3*(t2-t1):.0f} ms, {c}x{r} cells', file=sys.stderr)
        return 0

    cells_fit = V.Cells(a.set, cw, ch, k=a.k)
    t2 = time.perf_counter()
    c, r = V.fit_size(src.shape, cols, a.rows, cw, ch)
    img = V.resample(src, c * cw, r * ch, linear=not a.srgb_resample)
    cells = cells_fit.fit(img)
    t3 = time.perf_counter()
    sys.stdout.write(V.ansi_rows(cells, cells_fit.scalars))
    sys.stdout.flush()
    if a.preview or a.stats:
        drawn = cells_fit.decode(cells)
        if a.preview:
            V.png(a.preview, drawn)
        if a.stats:
            import numpy as np
            mse = ((drawn.astype(float) - img) ** 2).mean()
            psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
            print(f'decode {1e3*(t1-t0):.0f} ms, masks {1e3*(t2-t1):.0f} ms ({len(cells_fit.scalars)} candidates), '
                  f'fit {1e3*(t3-t2):.0f} ms, {c}x{r} cells, PSNR {psnr:.2f} dB', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
