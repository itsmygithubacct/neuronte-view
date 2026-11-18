# neuronte-view

Show a picture (JPEG, PNG, WebP, GIF and anything else Pillow opens) or a PDF
page in a terminal, drawn by Neuronte's nets.

This is not part of Neuronte. It depends on a built neuronte-v2 tree, named
by `NEURONTE_HOME` or found where Kilix installs it
(`~/.local/gpu_terminal/sources/neuronte`). From that tree it uses
`tools/neuronte_video.py`, the weight files and `build/libneuronte.so`, and
it copies none of them.

**Fonts.** Neuronte's own claim (no font file is opened by its tools) does
not cover this program:

- PDF pages are drawn with poppler (`gi.repository.Poppler`, `pdftocairo`),
  and poppler loads the fonts embedded in the PDF, or system fonts in their
  place.
- `tests/check_view.py` writes its test PDF with cairo, which uses a system
  font.
- `open_pdf_page_neuronte()` leaves the PDF's own text out of the page and
  draws it with Neuronte instead. Even so, poppler still reads the PDF's
  fonts to make the SVG it starts from.

## Use

```sh
export NEURONTE_HOME=/path/to/neuronte-v2      # built: make build/libneuronte.so
./neuronte-view.py photo.jpg                   # cells, terminal width
./neuronte-view.py --cols 100 --set full photo.jpg
./neuronte-view.py --mode pixels photo.jpg     # kitty graphics protocol
./neuronte-view.py --page 2 paper.pdf
./neuronte-view.py --preview out.png --stats photo.jpg
```

Needs Python 3 with numpy and Pillow. PDF pages also need poppler's GObject
bindings, pycairo, librsvg's GObject bindings and `pdftocairo`. On Debian
these are python3-numpy, python3-pil, gir1.2-poppler-0.18, python3-cairo,
gir1.2-rsvg-2.0 and poppler-utils.

## Cells

Each cell gets one character, a foreground colour and a background colour.

- **Where the shapes come from:** every candidate character is drawn by the
  bundles a Neuronte terminal draws with. They are routed as neuroterm routes
  them: the newest `neuronte-split-vN.nrt`, with characters it was not
  trained on passed on, then every `neuronte-term-*.nrt` by name.
- **Cell size:** characters are drawn at the terminal's cell size
  (`--cell`, default 10x20). So the fit is judged on the pixels a Neuronte
  terminal shows.
- **Cache:** masks are cached in `~/.cache/neuronte-view`, keyed by the
  weights' SHA-256 and the cell size. Identical masks are kept once.

The candidate sets (`--set`):

| Set | Characters |
|---|---|
| `pixel` | space, blocks, quadrants, sextants, octants |
| `draw` (default) | `pixel`, plus box drawing, shades, Braille, and the triangles and quarter circles a term bundle draws |
| `full` | every one-cell character the bundles draw, about 4,600 |

How a cell is fitted:

- **Ranking:** `TwoInk.rank` orders every candidate by its exact
  unconstrained least-squares error, using one matrix product. It is in
  neuronte-v2 after 2.5.0; for older trees this program carries the same
  function.
- **Fit:** the top `--k` candidates (default 12), plus the flat mask, get
  the bounded fit.
- **Resampling:** pictures are resampled with Lanczos in linear light.

## Pixels

`--mode pixels` draws the picture through `MultiInk`: the three neural masks
of `neuronte-video-masks-v0.nrt`, at 2x4 cells. It sends the result with the
kitty graphics protocol, which Kilix supports.

## PDF pages without the PDF's text faces

`open_pdf_page_neuronte()` starts from poppler's SVG of the page and removes
every glyph from it, so pictures and vector drawing stay. It then draws each
character with Neuronte's proportional face for its size: hand-drawn up to
16 px, scaled to 32 px, large above. Each character goes at the place,
baseline, colour and weight the page gives it.

## Tests

```sh
NEURONTE_HOME=... python3 tests/check_view.py
```

The tests check:

- the ranking against a brute-force search;
- a full shortlist against no shortlist;
- that every candidate is drawn by a bundle;
- the PDF path.

## Measured

A comparison with tiv and chafa, every tool's output drawn by Neuronte at
10x20, is in the author's research notes. Over 10 images at 80 columns:

| Tool | PSNR after a half-cell blur (dB) |
|---|---:|
| neuronte-view, `draw` set | 40.9 |
| chafa | 37.8 |
| tiv | 35.1 |
