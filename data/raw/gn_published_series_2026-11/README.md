# Published group-number series (2026-11)

**Comparison only.** The firewall validator refuses these as a calibration input
(ADR-002, ADR-003) and the Step 2 scorer never reads them. They are here so the
benchmark's baseline scores can be read beside the real series the same methods
produced.

Exported by `scripts/fetch_published_gn_series.py` from the read-only 2017
MATLAB snapshot in `legacy/GN_Series_matlab/data/`, **not** re-downloaded: those
are the exact bytes the legacy analyses used, which makes a comparison against
the legacy baselines like for like. `provenance.json` carries each source file's
SHA-256.

| series | years | rows | reference |
|---|---|---|---|
| `hoyt_schatten_1998` | 1610-1995 | 386 | Hoyt & Schatten 1998, Sol. Phys. 179, 189 |
| `svalgaard_schatten_2016_backbone` | 1610-2015 | 406 | Svalgaard & Schatten 2016, Sol. Phys. 291, 2653 |
| `usoskin_series` | 1749-1995 | 247 | Usoskin et al., active-day-fraction family (exact version not recorded upstream) |
| `vaquero` | 1639-1714 | 26 | Vaquero et al., as carried by the 2017 snapshot |
| `zolotova` | 1635-1720 | 127 | Zolotova & Ponyavin, as carried by the 2017 snapshot |

## Not acquired

- **chatzistergos_2017** -- A&A 602, A69 supplement at CDS/VizieR -- unreachable from this machine
- **velasco_herrera_2024** -- Adv. Space Res. 73, 2788 supplement -- no open source found

Both are recorded in `docs/04_data/refresh_todo.md`. The benchmark does not
depend on them.

## A caveat on the Usoskin series

The snapshot's `GNiu_y2.txt` carries no version string, so it is recorded as
"Usoskin et al., active-day-fraction family" and **not** asserted to be Usoskin
et al. 2021. Do not cite it as a specific publication without checking.
