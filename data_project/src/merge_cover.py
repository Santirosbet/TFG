"""
Merge BIDA cover sheet as first pages of TFG_Final_v7.docx.

Steps:
1. Copy unpacked_v6 → unpacked_final
2. Copy cover logo (image1.jpeg) → thesis media as image_cover.jpeg
3. Add rId_cover relationship to thesis rels
4. Add .jpeg content type if missing
5. Extract cover body content (lines 2 to last </w:p>)
   - Replace r:id="rId4" → r:id="rId_cover"
   - Insert <w:sectPr> inside last paragraph's <w:pPr> (before <w:rPr>)
6. Prepend cover content to thesis body
7. Pack as TFG_Final_v7_final.docx

Run: python src/merge_cover.py
"""

import os, shutil, subprocess, sys, zipfile, re
sys.stdout.reconfigure(encoding='utf-8')

DOCX_BASE = r"C:\Users\santi\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\644fd433-e47e-45fc-a2e4-186ff94b9414\e53c41d9-1c86-4563-85dd-500dc1c93ca4\skills\docx"
SCRIPTS   = os.path.join(DOCX_BASE, "scripts", "office")
UNPACKED_V6   = os.path.join(DOCX_BASE, "unpacked_v6")
UNPACKED_COVER = os.path.join(DOCX_BASE, "unpacked_cover")
OUT_DIR   = r"C:\Users\santi\OneDrive\Desktop\TFG\data_project\output"
ORIGINAL_DOCX = os.path.join(OUT_DIR, "TFG_Final_v7.docx")


def pack_docx(unpacked_dir, out_path, original_docx):
    pack_script = os.path.join(SCRIPTS, "pack.py")
    cmd = ["python", pack_script, unpacked_dir, out_path,
           "--original", original_docx, "--validate", "false"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=DOCX_BASE)
    if result.returncode != 0:
        print(f"  Pack error: {result.stderr[:500]}")
    else:
        print(f"  Packed: {out_path}")
    return result.returncode == 0


def main():
    # ── 1. Copy unpacked_v6 → unpacked_final ─────────────────────────────────
    unpacked_final = os.path.join(DOCX_BASE, "unpacked_final")
    if os.path.exists(unpacked_final):
        shutil.rmtree(unpacked_final)
    shutil.copytree(UNPACKED_V6, unpacked_final)
    print(f"Copied unpacked_v6 → unpacked_final")

    # ── 2. Copy cover logo ────────────────────────────────────────────────────
    cover_logo_src = os.path.join(UNPACKED_COVER, "word", "media", "image1.jpeg")
    cover_logo_dst = os.path.join(unpacked_final, "word", "media", "image_cover.jpeg")
    shutil.copy(cover_logo_src, cover_logo_dst)
    print(f"Copied cover logo → image_cover.jpeg")

    # ── 3. Add rId_cover relationship ─────────────────────────────────────────
    rels_path = os.path.join(unpacked_final, "word", "_rels", "document.xml.rels")
    with open(rels_path, "r", encoding="utf-8") as f:
        rels_content = f.read()
    new_rel = ('  <Relationship Id="rId_cover" '
               'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
               'Target="media/image_cover.jpeg"/>\n')
    rels_content = rels_content.replace("</Relationships>", new_rel + "</Relationships>")
    with open(rels_path, "w", encoding="utf-8") as f:
        f.write(rels_content)
    print("Added rId_cover to document.xml.rels")

    # ── 4. Add .jpeg content type if missing ──────────────────────────────────
    ct_path = os.path.join(unpacked_final, "[Content_Types].xml")
    with open(ct_path, "r", encoding="utf-8") as f:
        ct_content = f.read()
    if 'Extension="jpeg"' not in ct_content and 'Extension="jpg"' not in ct_content:
        jpeg_type = '<Default Extension="jpeg" ContentType="image/jpeg"/>\n'
        ct_content = ct_content.replace("</Types>", jpeg_type + "</Types>")
        with open(ct_path, "w", encoding="utf-8") as f:
            f.write(ct_content)
        print("Added jpeg content type")
    else:
        print("jpeg content type already present")

    # ── 5. Read and process cover body content ─────────────────────────────────
    cover_xml_path = os.path.join(UNPACKED_COVER, "word", "document.xml")
    with open(cover_xml_path, "r", encoding="utf-8") as f:
        cover_lines = f.readlines()

    total_cover = len(cover_lines)
    print(f"Cover document.xml: {total_cover} lines")

    # Find body-level <w:sectPr> (not inside a pPr - appears near end of body)
    # Scan from the end
    body_secpr_start = None
    body_secpr_end   = None
    for i in range(total_cover - 1, -1, -1):
        stripped = cover_lines[i].strip()
        if stripped == '</w:sectPr>' and body_secpr_end is None:
            body_secpr_end = i
        if body_secpr_end is not None and stripped.startswith('<w:sectPr') and body_secpr_start is None:
            body_secpr_start = i
            break

    if body_secpr_start is None:
        print("ERROR: Could not find body-level <w:sectPr>")
        return

    print(f"Body-level sectPr: lines {body_secpr_start+1} to {body_secpr_end+1} (1-indexed)")

    # sectPr XML to embed (strip trailing newlines for clean insertion)
    sectpr_lines = cover_lines[body_secpr_start:body_secpr_end + 1]
    # Re-indent sectPr content to fit inside pPr (add 2 spaces extra indent)
    sectpr_xml = ''
    for line in sectpr_lines:
        sectpr_xml += '  ' + line  # add 2 extra spaces

    # Cover body content = lines between <w:body> (index 1) and sectPr start
    # Find where <w:body> is
    body_start_idx = None
    for i, line in enumerate(cover_lines):
        if '<w:body>' in line:
            body_start_idx = i
            break
    if body_start_idx is None:
        print("ERROR: Could not find <w:body>")
        return

    # Extract cover body content (between <w:body> and sectPr)
    cover_body_lines = cover_lines[body_start_idx + 1 : body_secpr_start]
    cover_body_str = ''.join(cover_body_lines)

    # Update image rId reference (VML format uses r:id)
    cover_body_str = cover_body_str.replace('r:id="rId4"', 'r:id="rId_cover"')

    # Insert <w:sectPr> inside the last paragraph's <w:pPr>, AFTER </w:rPr>.
    # Schema order in pPr: ... ind → rPr → sectPr → pPrChange (optional)
    # The last paragraph has:
    #   <w:rPr>
    #     <w:rFonts .../>
    #     <w:sz .../>
    #   </w:rPr>
    # </w:pPr>
    # We insert sectPr after </w:rPr> and before </w:pPr>
    target = ('        </w:rPr>\n'
              '      </w:pPr>\n'
              '      <w:r>\n'
              '        <w:rPr>\n'
              '          <w:rFonts w:ascii="Arial" w:hAnsi="Arial"/>\n'
              '          <w:sz w:val="24"/>\n'
              '        </w:rPr>\n'
              '        <w:t>PRESIDENT OF THE PANEL</w:t>\n')
    replacement = ('        </w:rPr>\n'
                   + sectpr_xml
                   + '      </w:pPr>\n'
                   + '      <w:r>\n'
                   + '        <w:rPr>\n'
                   + '          <w:rFonts w:ascii="Arial" w:hAnsi="Arial"/>\n'
                   + '          <w:sz w:val="24"/>\n'
                   + '        </w:rPr>\n'
                   + '        <w:t>PRESIDENT OF THE PANEL</w:t>\n')

    if target in cover_body_str:
        cover_body_str = cover_body_str.replace(target, replacement)
        print("Inserted sectPr after </w:rPr> in last paragraph's pPr (correct schema order)")
    else:
        # Fallback: insert before last </w:pPr>
        last_ppr_close = cover_body_str.rfind('      </w:pPr>')
        if last_ppr_close != -1:
            cover_body_str = (cover_body_str[:last_ppr_close]
                              + sectpr_xml
                              + cover_body_str[last_ppr_close:])
            print("Inserted sectPr before last </w:pPr> (fallback)")
        else:
            print("WARNING: Could not find insertion point for sectPr")

    # ── 6. Read thesis document.xml and prepend cover content ─────────────────
    thesis_xml_path = os.path.join(unpacked_final, "word", "document.xml")
    with open(thesis_xml_path, "r", encoding="utf-8") as f:
        thesis_lines = f.readlines()

    print(f"Thesis document.xml: {len(thesis_lines)} lines")

    # Structure: line 0 = xml declaration + <w:document...>
    #            line 1 = <w:body>
    #            lines 2+ = body content
    # Find <w:body> line
    thesis_body_idx = None
    for i, line in enumerate(thesis_lines):
        if '<w:body>' in line:
            thesis_body_idx = i
            break
    if thesis_body_idx is None:
        print("ERROR: Could not find <w:body> in thesis")
        return

    print(f"Thesis <w:body> at index {thesis_body_idx}")

    # Build merged document
    merged_lines = (
        thesis_lines[:thesis_body_idx + 1]  # up to and including <w:body>
        + [cover_body_str]                   # cover content as single string
        + thesis_lines[thesis_body_idx + 1:] # rest of thesis body
    )

    with open(thesis_xml_path, "w", encoding="utf-8") as f:
        f.writelines(merged_lines)

    print(f"Merged document.xml written: {len(merged_lines)} items")

    # ── 6b. Fix invalid paraId values (CA00100x >= 0x80000000) ───────────────
    # These were assigned to table captions in unpacked_v6. Replace with valid
    # unique hex IDs < 0x7FFFFFFF that don't conflict with existing IDs.
    with open(thesis_xml_path, "r", encoding="utf-8") as f:
        merged_content = f.read()

    # Find all existing paraId values to avoid collisions
    existing_ids = set(re.findall(r'w14:paraId="([0-9A-F]{8})"', merged_content, re.IGNORECASE))
    print(f"Found {len(existing_ids)} existing paraIds in merged document")

    # Generate valid replacement IDs starting from 0x5A000001
    invalid_ids_pattern = re.compile(r'(w14:paraId=")(CA00[0-9A-F]{4})"', re.IGNORECASE)
    invalid_ids_found = set(m.group(2) for m in invalid_ids_pattern.finditer(merged_content))
    print(f"Invalid paraIds to replace: {invalid_ids_found}")

    counter = 0x5A000001
    id_map = {}
    for bad_id in sorted(invalid_ids_found):
        new_id = f"{counter:08X}"
        while new_id.upper() in {i.upper() for i in existing_ids}:
            counter += 1
            new_id = f"{counter:08X}"
        id_map[bad_id.upper()] = new_id
        existing_ids.add(new_id)
        counter += 1

    print(f"Replacement map: {id_map}")

    # Replace each occurrence (each needs a unique ID, so replace one at a time)
    for bad_id, new_id in id_map.items():
        # Replace the first occurrence with the new unique ID,
        # but since there may be duplicates, replace ALL occurrences of this bad_id
        merged_content = re.sub(
            r'w14:paraId="' + re.escape(bad_id) + '"',
            f'w14:paraId="{new_id}"',
            merged_content,
            flags=re.IGNORECASE
        )

    # Note: if CA001003 appeared twice (as a bug), both get the same new ID.
    # To handle that, do a second pass to de-duplicate any remaining paraIds.
    # Find all paragraphs with their paraIds and ensure uniqueness.
    all_para_ids = re.findall(r'w14:paraId="([0-9A-F]{8})"', merged_content, re.IGNORECASE)
    seen_ids = set()
    dup_counter = 0x5B000001
    def replace_dup(m):
        nonlocal dup_counter
        pid = m.group(1).upper()
        if pid in seen_ids:
            # Find a new unique ID
            new = f"{dup_counter:08X}"
            while new.upper() in {i.upper() for i in seen_ids}:
                dup_counter += 1
                new = f"{dup_counter:08X}"
            seen_ids.add(new.upper())
            dup_counter += 1
            return f'w14:paraId="{new}"'
        seen_ids.add(pid)
        return m.group(0)

    merged_content = re.sub(
        r'w14:paraId="([0-9A-F]{8})"',
        replace_dup,
        merged_content,
        flags=re.IGNORECASE
    )

    with open(thesis_xml_path, "w", encoding="utf-8") as f:
        f.write(merged_content)
    print("Fixed invalid paraIds and de-duplicated all paraIds")

    # ── 7. Pack final docx ────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "TFG_Final_v7_final.docx")
    success = pack_docx(unpacked_final, out_path, ORIGINAL_DOCX)

    if success:
        # Quick word count check
        with zipfile.ZipFile(out_path) as z:
            xml = z.read('word/document.xml').decode('utf-8')
        texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', xml)
        wc = len(' '.join(texts).split())
        print(f"\n=== DONE ===")
        print(f"  Output: {out_path}")
        print(f"  Total words: {wc:,}")
    else:
        print("\n=== PACK FAILED ===")


if __name__ == "__main__":
    main()
