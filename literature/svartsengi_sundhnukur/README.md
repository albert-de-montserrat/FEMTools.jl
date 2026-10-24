# Svartsengi / Sundhnúkur 2023–25

- `Parks2025_EPSL_Svartsengi.pdf`: Parks et al. (2025), EPSL 658, 119324,
  doi:10.1016/j.epsl.2025.119324 (open-access copy, White Rose eprints 229677).
  Data statement: "Data will be made available on request."
- `EGU25-17523_abstract.html`: Sigmundsson et al., EGU25 abstract.

## Digitized from Parks et al. (2025) Fig. 6 (300-dpi raster, PDF page 9)

Volume change and its rate in the Svartsengi magma domain from GNSS, Mogi
point source at 4 km depth (−22.454° E, 63.869° N).

- `Parks2025_fig6a_volume_daily.csv`: `segment, date, volume_change_1e6m3`,
  daily median of the marker ring centers; `segment` counts the
  inter-event intervals (1 = before the 10 Nov 2023 dike).
- `Parks2025_fig6b_rates.csv`: the 14 rate bars, `start_date, end_date,
  rate_m3_per_s`.
- Axes calibrated on tick marks: ~0.13 ×10⁶ m³ per pixel, ~0.37 day per
  pixel. The event lines sit ~1 day after the event dates on this axis
  calibration, so dates carry ~±1 day.
- Least-squares slopes of panel (a) over each panel-(b) interval reproduce
  the bars to 0–4 % (≤1 % from May 2024).
- `digitize/`: the scripts (PNGFiles, Colors); input is
  `pdfimages -f 9 -l 9 -png` of the PDF.
