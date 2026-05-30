"""
Build TFG versions for the 10,000-word limit requirement.

Version 3 (Otilio's approach): Move excess sections to new appendices.
  - Main body ≈ 10,000 words
  - All content preserved in appendices B-I
  - Output: TFG_Final_v7_otilio.docx

Version 1 (Condensed): Same trimmed main body, no extra appendices.
  - Main body ≈ 10,000 words, truly standalone
  - Output: TFG_Final_v7_condensed.docx

v7 base: unpacked_v6 with table captions (Tabla 3-8) + in-text citations added.
Line numbers updated to match modified document.xml (19,518 lines).

Run: python src/build_thesis_versions.py
"""

import os, shutil, subprocess, textwrap, sys
sys.stdout.reconfigure(encoding='utf-8')

DOCX_BASE = r"C:\Users\santi\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\644fd433-e47e-45fc-a2e4-186ff94b9414\e53c41d9-1c86-4563-85dd-500dc1c93ca4\skills\docx"
SCRIPTS   = os.path.join(DOCX_BASE, "scripts", "office")
UNPACKED_SRC = os.path.join(DOCX_BASE, "unpacked_v6")
ORIGINAL_DOCX = r"C:\Users\santi\OneDrive\Desktop\TFG\data_project\output\TFG_Final_v7.docx"
OUT_DIR   = r"C:\Users\santi\OneDrive\Desktop\TFG\data_project\output"

# ─── Section boundaries (1-indexed line numbers in unpacked document.xml) ────
# Each entry: (section_name, para_start_line, para_end_line_exclusive,
#              appendix_letter, appendix_title, cross_ref_note)
MOVE_SECTIONS = [
    (
        "Section 3 — Background",
        1441, 1657,
        "B",
        "Appendix B — Theoretical and Conceptual Framework",
        "Note: The complete Theoretical and Conceptual Framework (Section 3 of the "
        "original document) is provided in Appendix B, covering pharmaceutical supply "
        "chain challenges, the Bullwhip effect, time-series forecasting theory, and "
        "Business Intelligence foundations."
    ),
    (
        "Section 6.11 — Southern Hemisphere Validation",
        10900, 13218,
        "C",
        "Appendix C — Southern Hemisphere Multi-Country Validation (Section 6.11)",
        "Note: The full multi-country Southern Hemisphere cross-hemispheric validation "
        "study (Section 6.11) — including CCF analysis for 7 countries (Australia, "
        "New Zealand, Chile, Argentina, South Africa, Brazil, Uruguay) and XGBoost "
        "model comparisons — is provided in Appendix C. Key finding: all non-tropical "
        "SH countries show significant positive lead-lag correlation (r > 0.26, "
        "p < 0.001), confirming a hemispheric mechanism rather than an "
        "Australia-specific coincidence."
    ),
    (
        "Section 6.13 — Ensemble and Switching Rule (detailed)",
        13218, 14173,
        "D",
        "Appendix D — Advanced Experiments: Ensemble and Switching Rule (Section 6.13)",
        "Note: Full methodology and results for the multi-country SH ensemble and "
        "seasonal switching rule experiments are provided in Appendix D. Summary: "
        "the seasonal switching rule (replace XGBoost with historical seasonal mean "
        "during off-season weeks 22-39) achieves 35.78% MAPE — the best result in "
        "this thesis, a 8.37 percentage-point improvement over standalone XGBoost."
    ),
    (
        "Section 6.14 — Prophet Baseline",
        14173, 14957,
        "E",
        "Appendix E — Prophet Baseline: Third Model Comparison (Section 6.14)",
        "Note: Complete Prophet baseline analysis (Section 6.14) is provided in "
        "Appendix E. Summary: Prophet achieves 48.32% MAPE — worse than XGBoost B "
        "(44.16%) but comparable in MAE. Diebold-Mariano test confirms XGBoost B "
        "significantly outperforms Prophet (DM = 2.14, p = 0.034)."
    ),
    (
        "Section 6.19 — TFG v2 External Signal Experiments",
        14957, 15277,
        "F",
        "Appendix F — TFG Version 2: External Signal Extensions (Section 6.19)",
        "Note: TFG Version 2 experiment results (Section 6.19) are provided in "
        "Appendix F. These include LightGBM, stacking ensemble, conformal prediction, "
        "Google Trends, and temperature signals — all documented as negative results "
        "(none improve on XGBoost B MAPE). Their inclusion demonstrates rigorous "
        "experimental validation."
    ),
    (
        "Section 6.20 — TFG v3 Feature Engineering",
        15277, 15427,
        "G",
        "Appendix G — TFG Version 3: Feature Engineering and Cross-Category (Section 6.20)",
        "Note: TFG Version 3 experiments (Section 6.20) are provided in Appendix G, "
        "including enhanced feature engineering (cyclical week encoding, multi-lag AR) "
        "and N02BE (paracetamol) cross-category validation — confirming framework "
        "generalisability across therapeutic categories."
    ),
    (
        "Section 6.21 — Robustness Analysis",
        15427, 15596,
        "H",
        "Appendix H — Robustness Analysis: Indirect Evidence Under Anomalous Conditions (Section 6.21)",
        "Note: Full robustness analysis (Section 6.21) is provided in Appendix H, "
        "covering season severity stratification, directional accuracy (75.8%), "
        "temperature confounding test (r = -0.151), bootstrap MAPE stability "
        "(bootstrap mean 40.4%, 95% CI [35.1%, 46.2%]), and 2017-18 case study. "
        "All five analyses confirm system stability under non-standard conditions."
    ),
    (
        "Section 6.22 — Feature-Enriched Backtest",
        15596, 16438,
        "I",
        "Appendix I — Feature-Enriched Model: Season-by-Season Backtest (Section 6.22)",
        "Note: Full season-by-season backtest validation for the feature-enriched "
        "XGBoost model (Section 6.22) is provided in Appendix I, including architecture "
        "details (32 features), protocol, and results across three European flu seasons "
        "(2016-17: 42.3% MAPE; 2017-18: 37.9%; 2018-19: 38.1%; mean 39.4%)."
    ),
]

# ─── Helper: build a cross-reference paragraph in Word XML ────────────────────
def make_xref_para(text):
    """Return XML for an italic note paragraph."""
    # Escape XML special chars
    text = (text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                .replace('"',"&quot;"))
    return (
        '    <w:p w14:paraId="BB000001" w14:textId="77777777" '
        'w:rsidR="00715D74" w:rsidRDefault="00715D74">\n'
        '      <w:pPr>\n'
        '        <w:spacing w:before="80" w:after="160"/>\n'
        '        <w:rPr><w:i/><w:sz w:val="20"/></w:rPr>\n'
        '      </w:pPr>\n'
        '      <w:r>\n'
        '        <w:rPr><w:i/><w:sz w:val="20"/></w:rPr>\n'
        f'        <w:t>{text}</w:t>\n'
        '      </w:r>\n'
        '    </w:p>\n'
    )


def make_app_heading(letter, title):
    """Return XML for a Ttulo2 appendix heading."""
    title = title.replace("&","&amp;")
    return (
        '    <w:p w14:paraId="BB000002" w14:textId="77777777" '
        'w:rsidR="00715D74" w:rsidRDefault="00715D74">\n'
        '      <w:pPr>\n'
        '        <w:pStyle w:val="Ttulo2"/>\n'
        '      </w:pPr>\n'
        '      <w:r>\n'
        '        <w:rPr><w:color w:val="2E75B6"/></w:rPr>\n'
        f'        <w:t>{title}</w:t>\n'
        '      </w:r>\n'
        '    </w:p>\n'
    )


def pack_docx(unpacked_dir, out_path, original_docx):
    """Pack an unpacked dir into a docx using the skill's pack.py."""
    pack_script = os.path.join(SCRIPTS, "pack.py")
    cmd = ["python", pack_script, unpacked_dir, out_path,
           "--original", original_docx, "--validate", "false"]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=DOCX_BASE)
    if result.returncode != 0:
        print(f"  Pack error: {result.stderr[:300]}")
    else:
        print(f"  Packed: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    doc_xml_path = os.path.join(UNPACKED_SRC, "word", "document.xml")
    with open(doc_xml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Loaded document.xml: {total} lines")

    # Verify section boundaries
    print("\nVerifying section boundaries...")
    for name, start, end, *_ in MOVE_SECTIONS:
        print(f"  {name}: lines {start}-{end} ({end-start} lines)")

    # ── Build the modified main body (shared between V1 and V3) ──────────────
    # Create a list of line ranges to KEEP, with replacements at cut points
    keep_ranges = []  # list of (start, end) inclusive 1-indexed lines to keep
    replacements = {}  # line_after_cut -> replacement text to insert

    sorted_moves = sorted(MOVE_SECTIONS, key=lambda x: x[1])  # sort by start line

    prev_end = 1
    for name, start, end, letter, app_title, xref_note in sorted_moves:
        # Keep everything from prev_end to start-1
        if prev_end <= start - 1:
            keep_ranges.append((prev_end, start - 1))
        # Insert cross-reference after cut
        replacements[start] = make_xref_para(xref_note)
        prev_end = end

    # Keep the rest of the document
    keep_ranges.append((prev_end, total))

    # Build modified lines (for main body)
    modified_lines = []
    for (keep_start, keep_end) in keep_ranges:
        # Add cross-reference before this range if we just skipped a section
        if keep_start in replacements:
            modified_lines.append(replacements[keep_start])
        modified_lines.extend(lines[keep_start - 1 : keep_end])

    print(f"\nModified main body: {len(modified_lines)} lines "
          f"(was {total}, removed {total - len(modified_lines)})")

    # ── VERSION 1: Condensed — no extra appendices ────────────────────────────
    print("\n── Building Version 1: Condensed ──")
    v1_dir = os.path.join(DOCX_BASE, "unpacked_v8_condensed")
    if os.path.exists(v1_dir):
        shutil.rmtree(v1_dir)
    shutil.copytree(UNPACKED_SRC, v1_dir)

    v1_xml = os.path.join(v1_dir, "word", "document.xml")
    with open(v1_xml, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)
    print(f"  Written {len(modified_lines)} lines to document.xml")

    v1_out = os.path.join(OUT_DIR, "TFG_Final_v7_condensed.docx")
    pack_docx(v1_dir, v1_out, ORIGINAL_DOCX)

    # ── VERSION 3: Otilio — main body + new appendices ────────────────────────
    print("\n── Building Version 3: Otilio appendix approach ──")
    v3_dir = os.path.join(DOCX_BASE, "unpacked_v7_otilio")
    if os.path.exists(v3_dir):
        shutil.rmtree(v3_dir)
    shutil.copytree(UNPACKED_SRC, v3_dir)

    # Find insertion point: just before </w:body> (last few lines)
    # In the modified main body, find the end of the existing appendices
    # We'll insert new appendix sections before the </w:body> tag

    # Build new appendix content
    appendix_xml = []
    for name, start, end, letter, app_title, xref_note in sorted_moves:
        appendix_xml.append(f'\n    <!-- ═══ {app_title} ═══ -->\n')
        appendix_xml.append(make_app_heading(letter, app_title))
        # Add the original section lines
        appendix_xml.extend(lines[start - 1 : end - 1])

    # Find the </w:body> in modified_lines
    insert_before = None
    for i in range(len(modified_lines) - 1, -1, -1):
        if '</w:body>' in modified_lines[i] or '<w:sectPr' in modified_lines[i]:
            insert_before = i
            break

    if insert_before is None:
        insert_before = len(modified_lines)
        print("  WARNING: Could not find </w:body>, appending at end")

    # Build final V3 lines
    v3_lines = (modified_lines[:insert_before]
                + appendix_xml
                + modified_lines[insert_before:])

    v3_xml = os.path.join(v3_dir, "word", "document.xml")
    with open(v3_xml, "w", encoding="utf-8") as f:
        f.writelines(v3_lines)
    print(f"  Written {len(v3_lines)} lines to document.xml")

    v3_out = os.path.join(OUT_DIR, "TFG_Final_v7_otilio.docx")
    pack_docx(v3_dir, v3_out, ORIGINAL_DOCX)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n═══ DONE ═══")
    print(f"  V1 Condensed:  {v1_out}")
    print(f"  V2 Full final: {ORIGINAL_DOCX} (already exists)")
    print(f"  V3 Otilio:     {v3_out}")

    # Quick word count of each
    import zipfile, re as _re
    for label, path in [("V1 condensed", v1_out), ("V3 otilio", v3_out)]:
        with zipfile.ZipFile(path) as z:
            xml = z.read('word/document.xml').decode('utf-8')
        texts = _re.findall(r'<w:t[^>]*>([^<]+)</w:t>', xml)
        wc = len(' '.join(texts).split())
        print(f"  {label}: {wc:,} words total")


if __name__ == "__main__":
    main()
