import ast
import sys

files = [
    "d:/RSNA/src_v2/dataset.py",
    "d:/RSNA/src_v2/train.py",
    "d:/RSNA/src_v2/model.py",
]
ok = True
for f in files:
    try:
        with open(f, encoding="utf-8") as fh:
            source = fh.read()
        ast.parse(source)
        print(f"OK: {f}")
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {f}: {e}")
        ok = False

sys.exit(0 if ok else 1)
