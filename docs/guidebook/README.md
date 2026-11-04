# AgenticTO Guidebook (LaTeX source)

This folder holds the source of `AgenticTO_Guidebook.pdf`, the guide for reviewers and engineers.

## Layout

| Path | What it is |
|---|---|
| `guidebook.tex` | Main file: front matter, chapter order, index. |
| `agentto-guide.sty` | The book's style: fonts, colours, chapter openers, boxes, tables, listings, `\code`/`\tool` macros. |
| `chapters/` | One file per chapter and appendix, written by hand. |
| `generated/` | Reference tables generated from the code by `gen_tables.py`. Do not edit these by hand. |
| `gen_tables.py` | Generator for `generated/`. |
| `figures/` | Figures used only by the book (workflow figure, title art, one GUI screenshot). |
| `index.ist` | `makeindex` style (letter headings, dotted leaders). |
| `latexmkrc` | `latexmk` settings: pdfLaTeX, output in `build/`, index style. |
| `build/` | Build output (temporary). |

Most screenshots are read from `../user_guide/img/` (see `\graphicspath` in `guidebook.tex`).

## Rebuild

Run from this folder (`docs/guidebook/`).

1. Regenerate the reference tables. This imports `agentto.spec` and `agentto.tools` from `../../src` and reads other
   source files as text. It does not need the solver or a GPU, and it calls no language model.

   ```
   PYTHONIOENCODING=utf-8 python -B gen_tables.py
   ```

   Use the Python of the `agentto` environment (pydantic 2). The script stops with an error if a hand-written
   description in it no longer matches the code (for example a new rule, tool or certificate check), and it refuses to
   finish if any `.tex`, `.sty` or this README contains a local absolute path.

2. Compile with pdfLaTeX through `latexmk` (TeX Live or MiKTeX; packages include XCharter, newtx, tcolorbox, xltabular,
   listings, titlesec, imakeidx):

   ```
   latexmk guidebook.tex
   ```

   The PDF is written to `build/guidebook.pdf`. Copy it to `AgenticTO_Guidebook.pdf` in this folder.

3. To start from a clean state: `latexmk -C guidebook.tex`.

## What is generated from the code

Appendix A (specification reference, from the pydantic model), the rule tables and rule cards of Chapter 6 (from
`rules.py`), the tool tables of Chapter 9 and Appendix B (from the tool registry), the certificate checks and record,
report and solve fields (Chapter 7, Appendix C), the continuation schedules (Chapter 7), the MCP flags, the HTTP API
table, the keyboard table (Appendix D, from the web front end) and the script catalogue (Appendix E). The constants
used in the text (ladder limits, solver switch, iteration cap, number of tools per profile) are in
`generated/constants.tex`.

Everything else is written by hand. Numbers quoted from the paper's experiments carry the claim identifiers of
`results/claims_ledger.md` in grey brackets.
