
WIP based on my undertanding; **unofficial**

Not tested fully

make sure you are in a virtual env with pyarrow installed
```
python view_pack_web.py math-finemath.parquet
```

## Recent changes
- Hex dump lists every character (ASCII letters included), first 2000 chars.
- Selecting text and clicking `evidence` / `hex` appends each new selection (`; `-separated); duplicates are skipped.
