
WIP based on my undertanding; **unofficial**

Not tested fully

make sure you are in a virtual env with pyarrow installed
```
python view_pack_web.py math-finemath.parquet
```

## Recent changes
- Hex dump lists every character (ASCII letters included), first 2000 chars.
- Selecting text and clicking `evidence` / `hex` appends each new selection (`; `-separated); duplicates are skipped.
- Diseases keep the order they are ticked in (not sorted), so they line up with evidence/hex.
- Section A/B disease candidates are highlighted (B3 ghost markers in red) and listed; `+` ticks the code and appends evidence/hex. Toggle with `g`.
- Save sends only edited rows, so rows changed in labels.csv elsewhere are not overwritten.
- Save button turns orange with unsaved changes (count shown), grey once saved.
- Emptying a row (or "clear this row") removes it from labels.csv on save.
