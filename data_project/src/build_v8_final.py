"""
Build TFG_Final_v8_final.docx — submission-ready version.

Body target: ~10,000 words (sections 1-8 without bibliography/appendices).
Strategy: remove 12 sections from the body, move them to Appendices B-M,
          then prepend the BIDA cover sheet (2 pages).

Base document: unpacked_v6 (TFG_Final_v7.docx with all tutor comments applied)

Run: python src/build_v8_final.py
"""

import os, shutil, subprocess, sys, zipfile, re
sys.stdout.reconfigure(encoding='utf-8')

DOCX_BASE  = r"C:\Users\santi\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\644fd433-e47e-45fc-a2e4-186ff94b9414\e53c41d9-1c86-4563-85dd-500dc1c93ca4\skills\docx"
SCRIPTS    = os.path.join(DOCX_BASE, "scripts", "office")
UNPACKED_SRC   = os.path.join(DOCX_BASE, "unpacked_v6")
UNPACKED_COVER = os.path.join(DOCX_BASE, "unpacked_cover")
ORIGINAL_DOCX  = r"C:\Users\santi\OneDrive\Desktop\TFG\data_project\output\TFG_Final_v7.docx"
OUT_DIR    = r"C:\Users\santi\OneDrive\Desktop\TFG\data_project\output"

# ─── 12 sections to remove from body (sorted by start line, 1-indexed <w:p>) ─
# Rule: start/end = pStyle line − 2  (points to the opening <w:p> tag)
MOVE_SECTIONS = [
    (
        "Section 2.2 — Seasonal Influenza Epidemiology",
        1303, 1355,
        "J",
        "Appendix J — Seasonal Influenza Epidemiology and Cross-Hemispheric Lead-Lag",
        "Note: A detailed discussion of seasonal influenza epidemiology and the "
        "cross-hemispheric lead-lag signal that underlies the central hypothesis of "
        "this thesis — including WHO surveillance data, hemispheric transmission "
        "mechanisms, and the empirical basis for the 28-week lead time — is provided "
        "in Appendix J."
    ),
    (
        "Section 2.3 — ML Methods for Healthcare",
        1355, 1408,
        "K",
        "Appendix K — Machine Learning Methods for Pharmaceutical Demand Forecasting",
        "Note: A comprehensive review of machine learning methods applied to "
        "pharmaceutical and healthcare demand forecasting — including gradient "
        "boosting, ensemble methods, and conformal prediction — is provided in "
        "Appendix K, covering the literature that informed model selection in this "
        "thesis."
    ),
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
        "Section 6.0 — Research Journey",
        4494, 6237,
        "L",
        "Appendix L — Research Evolution: Iterative Data and Model Development",
        "Note: A full narrative of the iterative research process — including "
        "all intermediate model versions, data engineering decisions, and experimental "
        "dead-ends that shaped the final methodology — is documented chronologically "
        "in Appendix L."
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
        "Section 6.13 — Ensemble and Switching Rule",
        13218, 14173,
        "D",
        "Appendix D — Advanced Experiments: Ensemble and Switching Rule (Section 6.13)",
        "Note: Full methodology and results for the multi-country SH ensemble and "
        "seasonal switching rule experiments are provided in Appendix D. Summary: "
        "the seasonal switching rule achieves 35.78% MAPE — the best result in "
        "this thesis, an 8.37 percentage-point improvement over standalone XGBoost."
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
        "Section 6.19 — TFG v2 External Signals",
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
        "(bootstrap mean 40.4%, 95% CI [35.1%, 46.2%]), and 2017-18 case study."
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
    (
        "Section 7.3.1 — Dashboard Operational Pages",
        16525, 16635,
        "M",
        "Appendix M — Dashboard Implementation: Operational and Analytical Pages",
        "Note: A complete page-by-page description of the 21-page Streamlit "
        "decision-support dashboard — including each operational page, its inputs, "
        "outputs, and business use case — is provided in Appendix M."
    ),
]


# ─── Helper: XML builders ──────────────────────────────────────────────────────
def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;"))

def make_xref_para(text):
    return (
        '    <w:p w14:paraId="BB000001" w14:textId="77777777" '
        'w:rsidR="00715D74" w:rsidRDefault="00715D74">\n'
        '      <w:pPr>\n'
        '        <w:spacing w:before="80" w:after="160"/>\n'
        '        <w:rPr><w:i/><w:sz w:val="20"/></w:rPr>\n'
        '      </w:pPr>\n'
        '      <w:r>\n'
        '        <w:rPr><w:i/><w:sz w:val="20"/></w:rPr>\n'
        f'        <w:t>{esc(text)}</w:t>\n'
        '      </w:r>\n'
        '    </w:p>\n'
    )

def make_app_heading(letter, title):
    return (
        '    <w:p w14:paraId="BB000002" w14:textId="77777777" '
        'w:rsidR="00715D74" w:rsidRDefault="00715D74">\n'
        '      <w:pPr>\n'
        '        <w:pStyle w:val="Ttulo2"/>\n'
        '      </w:pPr>\n'
        '      <w:r>\n'
        '        <w:rPr><w:color w:val="2E75B6"/></w:rPr>\n'
        f'        <w:t>{esc(title)}</w:t>\n'
        '      </w:r>\n'
        '    </w:p>\n'
    )

def pack_docx(unpacked_dir, out_path):
    pack_script = os.path.join(SCRIPTS, "pack.py")
    cmd = ["python", pack_script, unpacked_dir, out_path,
           "--original", ORIGINAL_DOCX, "--validate", "false"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=DOCX_BASE)
    if result.returncode != 0:
        print(f"  Pack error: {result.stderr[:400]}")
    else:
        print(f"  Packed: {out_path}")
    return result.returncode == 0

def fix_para_ids(content):
    """Replace any paraId >= 0x80000000 with valid unique values and de-duplicate."""
    # Replace BB00000x and CA00100x invalid IDs
    invalid_pat = re.compile(r'w14:paraId="(BB0000[0-9A-F]{2}|CA001[0-9A-F]{3})"', re.IGNORECASE)
    existing = set(m.upper() for m in re.findall(r'w14:paraId="([0-9A-F]{8})"', content, re.IGNORECASE))
    id_map = {}
    counter = 0x5C000001
    for m in invalid_pat.finditer(content):
        bad = m.group(1).upper()
        if bad not in id_map:
            new_id = f"{counter:08X}"
            while new_id.upper() in existing:
                counter += 1
                new_id = f"{counter:08X}"
            id_map[bad] = new_id
            existing.add(new_id.upper())
            counter += 1
    for bad, new_id in id_map.items():
        content = re.sub(r'w14:paraId="' + re.escape(bad) + '"',
                         f'w14:paraId="{new_id}"', content, flags=re.IGNORECASE)

    # De-duplicate any remaining duplicate paraIds
    seen = set()
    dup_ctr = [0x5D000001]
    def dedup(m):
        pid = m.group(1).upper()
        if pid in seen:
            new = f"{dup_ctr[0]:08X}"
            while new.upper() in seen:
                dup_ctr[0] += 1
                new = f"{dup_ctr[0]:08X}"
            seen.add(new.upper())
            dup_ctr[0] += 1
            return f'w14:paraId="{new}"'
        seen.add(pid)
        return m.group(0)
    content = re.sub(r'w14:paraId="([0-9A-F]{8})"', dedup, content, flags=re.IGNORECASE)
    return content


def main():
    # ── 1. Load source document ───────────────────────────────────────────────
    doc_xml = os.path.join(UNPACKED_SRC, "word", "document.xml")
    with open(doc_xml, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)
    print(f"Source document.xml: {total} lines")

    sorted_moves = sorted(MOVE_SECTIONS, key=lambda x: x[1])

    # ── 2. Build modified body (sections removed, cross-refs inserted) ─────────
    keep_ranges = []
    replacements = {}
    prev_end = 1
    for name, start, end, letter, app_title, xref_note in sorted_moves:
        if prev_end <= start - 1:
            keep_ranges.append((prev_end, start - 1))
        replacements[start] = make_xref_para(xref_note)
        prev_end = end
    keep_ranges.append((prev_end, total))

    modified_lines = []
    for (ks, ke) in keep_ranges:
        if ks in replacements:
            modified_lines.append(replacements[ks])
        modified_lines.extend(lines[ks - 1: ke])

    print(f"Modified body: {len(modified_lines)} lines (removed {total - len(modified_lines)})")

    # ── 3. Build appendix content (B-M) ───────────────────────────────────────
    appendix_xml = []
    for name, start, end, letter, app_title, xref_note in sorted_moves:
        appendix_xml.append(make_app_heading(letter, app_title))
        appendix_xml.extend(lines[start - 1: end - 1])

    # ── 4. Build Otilio version (body + appendices) ───────────────────────────
    print("\nBuilding v8_otilio (body + appendices B-M)...")
    v8_dir = os.path.join(DOCX_BASE, "unpacked_v8_otilio")
    if os.path.exists(v8_dir):
        shutil.rmtree(v8_dir)
    shutil.copytree(UNPACKED_SRC, v8_dir)

    # Find insertion point: just before </w:body> (body-level only, not inline sectPr)
    insert_before = None
    for i in range(len(modified_lines) - 1, -1, -1):
        line = modified_lines[i] if isinstance(modified_lines[i], str) else ''
        if '</w:body>' in line:
            insert_before = i
            break
    if insert_before is None:
        # Fallback: look for body-level sectPr (must be at body indent level, 4 spaces)
        for i in range(len(modified_lines) - 1, -1, -1):
            line = modified_lines[i] if isinstance(modified_lines[i], str) else ''
            if line.startswith('    <w:sectPr') or line.startswith('  <w:sectPr'):
                insert_before = i
                break
    if insert_before is None:
        insert_before = len(modified_lines)
        print("  WARNING: could not find </w:body>, appending at end")

    v8_lines = modified_lines[:insert_before] + appendix_xml + modified_lines[insert_before:]

    v8_xml_path = os.path.join(v8_dir, "word", "document.xml")
    with open(v8_xml_path, "w", encoding="utf-8") as f:
        f.writelines(v8_lines)
    print(f"  Written {len(v8_lines)} lines")

    # ── 5. Add cover sheet to v8_otilio ───────────────────────────────────────
    print("\nAdding cover sheet...")

    # 5a. Copy cover logo
    cover_logo_src = os.path.join(UNPACKED_COVER, "word", "media", "image1.jpeg")
    cover_logo_dst = os.path.join(v8_dir, "word", "media", "image_cover.jpeg")
    shutil.copy(cover_logo_src, cover_logo_dst)

    # 5b. Add rId_cover relationship
    rels_path = os.path.join(v8_dir, "word", "_rels", "document.xml.rels")
    with open(rels_path, "r", encoding="utf-8") as f:
        rels = f.read()
    if 'rId_cover' not in rels:
        new_rel = ('  <Relationship Id="rId_cover" '
                   'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                   'Target="media/image_cover.jpeg"/>\n')
        rels = rels.replace("</Relationships>", new_rel + "</Relationships>")
        with open(rels_path, "w", encoding="utf-8") as f:
            f.write(rels)

    # 5c. Add jpeg content type if missing
    ct_path = os.path.join(v8_dir, "[Content_Types].xml")
    with open(ct_path, "r", encoding="utf-8") as f:
        ct = f.read()
    if 'Extension="jpeg"' not in ct and 'Extension="jpg"' not in ct:
        ct = ct.replace("</Types>",
                        '<Default Extension="jpeg" ContentType="image/jpeg"/>\n</Types>')
        with open(ct_path, "w", encoding="utf-8") as f:
            f.write(ct)

    # 5d. Read and process cover body content
    cover_xml_path = os.path.join(UNPACKED_COVER, "word", "document.xml")
    with open(cover_xml_path, "r", encoding="utf-8") as f:
        cover_lines = f.readlines()

    # Find body-level sectPr (scan from end)
    bsp_start = bsp_end = None
    for i in range(len(cover_lines) - 1, -1, -1):
        s = cover_lines[i].strip()
        if s == '</w:sectPr>' and bsp_end is None:
            bsp_end = i
        if bsp_end is not None and s.startswith('<w:sectPr') and bsp_start is None:
            bsp_start = i
            break

    sectpr_xml = ''.join('  ' + l for l in cover_lines[bsp_start:bsp_end + 1])

    # Find <w:body> in cover
    body_idx = next(i for i, l in enumerate(cover_lines) if '<w:body>' in l)
    cover_body = ''.join(cover_lines[body_idx + 1: bsp_start])
    cover_body = cover_body.replace('r:id="rId4"', 'r:id="rId_cover"')

    # Insert sectPr after </w:rPr> before </w:pPr> in last cover paragraph
    target = ('        </w:rPr>\n'
              '      </w:pPr>\n'
              '      <w:r>\n'
              '        <w:rPr>\n'
              '          <w:rFonts w:ascii="Arial" w:hAnsi="Arial"/>\n'
              '          <w:sz w:val="24"/>\n'
              '        </w:rPr>\n'
              '        <w:t>PRESIDENT OF THE PANEL</w:t>\n')
    repl = ('        </w:rPr>\n'
            + sectpr_xml
            + '      </w:pPr>\n'
            + '      <w:r>\n'
            + '        <w:rPr>\n'
            + '          <w:rFonts w:ascii="Arial" w:hAnsi="Arial"/>\n'
            + '          <w:sz w:val="24"/>\n'
            + '        </w:rPr>\n'
            + '        <w:t>PRESIDENT OF THE PANEL</w:t>\n')
    if target in cover_body:
        cover_body = cover_body.replace(target, repl)
        print("  sectPr inserted correctly (after </w:rPr>)")
    else:
        print("  WARNING: sectPr insertion target not found — appending at end of cover")

    # 5e. Prepend cover to thesis body
    with open(v8_xml_path, "r", encoding="utf-8") as f:
        thesis_lines_final = f.readlines()

    body_idx2 = next(i for i, l in enumerate(thesis_lines_final) if '<w:body>' in l)
    merged = (thesis_lines_final[:body_idx2 + 1]
              + [cover_body]
              + thesis_lines_final[body_idx2 + 1:])

    merged_str = ''.join(str(x) for x in merged)

    # 5f. Fix invalid paraIds
    merged_str = fix_para_ids(merged_str)

    with open(v8_xml_path, "w", encoding="utf-8") as f:
        f.write(merged_str)
    print("  Merged cover + thesis body written")

    # ── 6. Pack final docx ────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "TFG_Final_v8_final.docx")
    success = pack_docx(v8_dir, out_path)

    if success:
        with zipfile.ZipFile(out_path) as z:
            xml = z.read('word/document.xml').decode('utf-8')
        texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', xml)
        wc = len(' '.join(texts).split())
        print(f"\n{'='*50}")
        print(f"  Output:       {out_path}")
        print(f"  Total words:  {wc:,}")
        print(f"  Body target:  ~10,350 words (sections 1-8 only)")
        print(f"  Appendices:   B through M (12 appendices)")
        print(f"  Cover sheet:  BIDA (2 pages, La Salle logo)")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
