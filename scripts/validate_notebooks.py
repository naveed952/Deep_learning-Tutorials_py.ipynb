"""Validate notebooks without downloading data or executing training."""
import ast
from pathlib import Path
import nbformat
root=Path(__file__).resolve().parents[1]
for path in sorted((root/'notebooks').glob('*.ipynb')):
    book=nbformat.read(path,as_version=4);nbformat.validate(book)
    for i,cell in enumerate(book.cells):
        if cell.cell_type=='code':ast.parse(cell.source,filename=f'{path.name}:cell{i}')
    print('PASS',path.name)
for path in (root/'tutorial_support').glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
print('All notebooks and shared modules passed syntax/schema checks.')
