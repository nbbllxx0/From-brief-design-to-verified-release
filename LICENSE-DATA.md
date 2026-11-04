# Licenses of this repository

This repository uses three licenses.

- **Code: MIT License** ([`LICENSE`](LICENSE)). It covers the software: `src/`, `scripts/`, `web/`, `tests/`, and
  every other source-code file in the repository, including the scripts stored with the results, except
  `src/voxfem/`.
- **The voxfem solver: BSD 3-Clause License** ([`src/voxfem/LICENSE`](src/voxfem/LICENSE)). It covers the GPU
  finite-element and SIMP solver in `src/voxfem/`, copyright (c) 2026 Shaoliang Yang, Jun Wang and Yunsheng Wang.
- **Data and documentation: Creative Commons Attribution 4.0 International (CC BY 4.0)**,
  <https://creativecommons.org/licenses/by/4.0/>. It covers everything else: the briefs and reference readings
  (`briefs/`), the frozen data (`data/`), the result records and session logs (`results/`), the documentation
  (`docs/`), and the paper materials, tables and figures (`paper/`).

Copyright (c) 2026 Shaoliang Yang and Jun Wang.

When you reuse the data or documentation, cite the paper given in [`CITATION.cff`](CITATION.cff).

## Not covered

- **Third-party software.** The Python and JavaScript dependencies listed in `pyproject.toml`, the README and
  `web/package-lock.json` are not included in this repository; each keeps its own license.
- **Adapted benchmark problems.** Some briefs restate well-known benchmark problems in new words and units; the
  supplement of the paper cites their sources. No text, geometry or figures from those sources are copied here.
