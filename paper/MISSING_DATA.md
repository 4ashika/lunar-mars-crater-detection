# Missing data and reference verification

## TBD inventory

There are no unresolved numeric TBDs in the current paper tables. The experiment runner produced non-empty detection, validation, and mission-planner exports for the selected runs. The result tables are generated from those CSVs by `scripts/collect_results.py`.

The following scale caveats are visible in the tables and captions rather than hidden as TBD values:

- Mars has a calibrated `m_per_px` value on 1 of 60 tiles. The other 59 use the configured 5 m/pixel fallback. Mars pixel-to-metre summaries, class bins, tile footprints, and the Jezero mission example therefore include assumed-scale results.
- Mars depth estimates are low confidence for all detections by default because the images do not reliably encode cast-shadow geometry.
- Summed tile area is the sum of individual image footprints; tile overlap was not measured or removed.

## [VERIFY] inventory

- **Bradski (2000), “The OpenCV Library”: [VERIFY] page range.** Bibliographic sources disagree on the page span. The BibTeX entry intentionally omits pages rather than choosing an unsupported range. The journal, volume/issue, title, author, and year are retained.

Other reference metadata were checked against publisher, proceedings, institutional, or official mission pages available during preparation. The Bandeira paper was published in 2012 (after 2011 acceptance); the Kim--Muller item is cited as the 2004 ISPRS proceedings paper, which discusses image/terrain crater detection and ellipse-oriented processing. No DOI or page range was added where it could not be substantiated.

## Reproducibility notes

- Run `python paper/scripts/run_experiments.py` from the repository root to recompute outputs with the current code and dataset files.
- Run `python paper/scripts/collect_results.py` to regenerate `generated_tables.tex` from the exported CSVs.
- The held-out split includes four Moon validation tiles and 13 Mars validation/test tiles. Greedy one-to-one axis-aligned box matching uses IoU 0.5 and retains all YOLO labels.
- The figures are 300-dpi plots composed from real repository tiles, intermediate arrays, detector outputs, and planner outputs. The Jezero figure uses its actual exported label boxes and v2 detections.
