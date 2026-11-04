"""Extract the embedded raster images of one PDF page (for figure digitization); pypdf only.
usage: python pdf_images.py <pdf> <page_1based> <out_dir>"""
import sys
import os
from pypdf import PdfReader

pdf, page, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
pg = PdfReader(pdf).pages[page-1]
stem = os.path.splitext(os.path.basename(pdf))[0][:12]+f'_p{page}'
for i, im in enumerate(pg.images):
    fn = os.path.join(out, f'{stem}_img{i}_{im.name}')
    with open(fn, 'wb') as f:
        f.write(im.data)
    print(fn, im.image.size if im.image is not None else None)
