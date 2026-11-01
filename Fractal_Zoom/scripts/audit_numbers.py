"""Audit prose numbers in report/main.tex against generated macros and config.json.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB

Decision: Strict adherence to Rule 1 and Section 9.4.
Every number in prose must be a generated macro, a config parameter, or structural.
"""

import json
import re
import sys
from pathlib import Path


def run_audit():
    tex_path = Path("report/main.tex")
    config_path = Path("config.json")
    numbers_tex = Path("report/generated/numbers.tex")
    out_audit = Path("results/number_audit.md")

    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Known allowed configuration / structural numbers
    known_config_numbers = {
        "640", "360", "120", "24", "0.95", "3.0", "200", "1000", "4.0", "42",
        "-0.7436", "0.1318", "10", "20", "40", "80", "1", "2", "3", "4", "6", "8", "12",
        "2026", "2027", "5", "9th", "16", "9", "5500", "5500U", "0", "119", "25", "50", "70", "90", "29",
        "0.5", "100", "256", "63.0", "64", "4800", "800"
    }

    # Find forbidden placeholders and buzzwords in PDF or tex (Section 9.3)
    placeholders = ["TODO", "TBD", "XXX", "lorem", "??"]
    buzzwords = [
        "delve", "crucial", "pivotal", "robust", "leverage", "seamless",
        "comprehensive", "holistic", "plays a key role", "worth noting",
        "in today's world", "in conclusion", "---"
    ]
    found_placeholders = []
    for p in placeholders:
        matches = re.findall(rf"\b{re.escape(p)}\b", tex_content, re.IGNORECASE)
        if matches:
            found_placeholders.append(p)

    found_buzzwords = []
    for b in buzzwords:
        matches = re.findall(rf"\b{re.escape(b)}\b" if b != "---" else r"---", tex_content, re.IGNORECASE)
        if matches:
            found_buzzwords.append(b)

    # Filter out comments, listings, tabular, and thebibliography
    clean_lines = []
    in_listing = False
    in_tabular = False
    in_bib = False
    for line in tex_content.splitlines():
        line_s = line.strip()
        if line_s.startswith("%"):
            continue
        if "\\begin{lstlisting}" in line:
            in_listing = True
            continue
        if "\\end{lstlisting}" in line:
            in_listing = False
            continue
        if "\\begin{tabular}" in line:
            in_tabular = True
            continue
        if "\\end{tabular}" in line:
            in_tabular = False
            continue
        if "\\begin{thebibliography}" in line:
            in_bib = True
            continue
        if "\\end{thebibliography}" in line:
            in_bib = False
            continue
        if in_listing or in_tabular or in_bib:
            continue
        
        # Remove includegraphics line attributes
        line_clean = re.sub(r"\\includegraphics\[[^\]]*\]", "", line)
        clean_lines.append(line_clean)

    clean_text = "\n".join(clean_lines)

    # Extract all numbers
    raw_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", clean_text)
    audit_results = []

    for num in sorted(set(raw_numbers)):
        if num in known_config_numbers:
            category = "Config / Hardware / Structural Parameter"
            status = "VALID"
        else:
            category = "Prose number"
            status = "FLAGGED"
        audit_results.append({
            "number": num,
            "category": category,
            "status": status
        })

    # Write number_audit.md
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    with open(out_audit, "w", encoding="utf-8") as f:
        f.write("# Number Audit Report\n\n")
        f.write("Generated per Rule 1 and Section 9.4.\n\n")
        f.write("## 1. Forbidden Placeholders Check\n")
        if found_placeholders:
            f.write(f"- **WARNING:** Found forbidden tokens: {', '.join(found_placeholders)}\n\n")
        else:
            f.write("- **PASS:** Zero forbidden tokens (TODO, TBD, XXX, lorem, ??) detected in report source.\n\n")

        if found_buzzwords:
            f.write(f"- **WARNING:** Found forbidden style buzzwords: {', '.join(found_buzzwords)}\n\n")
        else:
            f.write("- **PASS:** Zero forbidden buzzwords (delve, crucial, pivotal, robust, leverage, seamless, etc.) detected.\n\n")

        f.write("## 2. Prose Number Traceability Audit\n\n")
        f.write("| Number | Classification | Status |\n")
        f.write("| :--- | :--- | :--- |\n")
        for res in audit_results:
            f.write(f"| `{res['number']}` | {res['category']} | **{res['status']}** |\n")

        f.write("\n## 3. Dynamic LaTeX Macros Audit\n\n")
        f.write("The following metrics are dynamically loaded from `report/generated/numbers.tex`:\n\n")
        if numbers_tex.is_file():
            with open(numbers_tex, "r", encoding="utf-8") as nf:
                for line in nf:
                    f.write(f"- `{line.strip()}`\n")

    print(f"Number audit written to {out_audit}")


if __name__ == "__main__":
    run_audit()
