from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML gallery for manual visual relevance annotation."
    )
    parser.add_argument(
        "--template-csv",
        default="outputs/manual_relevance/manual_relevance_template.csv",
        help="Manual relevance CSV exported by prepare_manual_relevance.py.",
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed",
        help="Processed image root used to resolve image paths in the CSV.",
    )
    parser.add_argument(
        "--output-html",
        default="outputs/manual_relevance/manual_relevance_gallery.html",
        help="Output HTML annotation gallery.",
    )
    return parser.parse_args()


def normalize_rel_path(path_text: str) -> str:
    return str(path_text or "").replace("\\", "/").lstrip("/")


def html_relative_image_path(output_html: Path, processed_root: Path, image_rel_path: str) -> str:
    target = processed_root / normalize_rel_path(image_rel_path)
    relative = os.path.relpath(target.resolve(), output_html.resolve().parent)
    return normalize_rel_path(str(relative))


def load_rows(csv_path: Path, output_html: Path, processed_root: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            item = dict(row)
            item["row_index"] = index
            item["query_img_src"] = html_relative_image_path(
                output_html,
                processed_root,
                item["query_processed_relative_path"],
            )
            item["candidate_img_src"] = html_relative_image_path(
                output_html,
                processed_root,
                item["candidate_processed_relative_path"],
            )
            rows.append(item)
    return rows


def build_html(rows: list[dict]) -> str:
    rows_json = json.dumps(rows, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Manual Relevance Annotation</title>
  <style>
    :root {{
      --border: #d0d7de;
      --muted: #57606a;
      --bg: #f6f8fa;
      --blue: #0969da;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #1f2328;
      background: white;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
      padding: 12px 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      font-size: 14px;
    }}
    button, select, input {{
      font: inherit;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: white;
      padding: 7px 10px;
    }}
    button {{
      cursor: pointer;
    }}
    button.primary {{
      background: var(--blue);
      color: white;
      border-color: var(--blue);
    }}
    .stats {{
      color: var(--muted);
      margin-left: auto;
    }}
    main {{
      padding: 16px;
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 14px;
      overflow: hidden;
    }}
    .card-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
      padding: 9px 12px;
      font-size: 13px;
    }}
    .pair {{
      display: grid;
      grid-template-columns: 1fr 1fr 240px;
      gap: 14px;
      padding: 12px;
      align-items: start;
    }}
    .image-block {{
      min-width: 0;
    }}
    .image-block h3 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    .image-block img {{
      display: block;
      width: 100%;
      max-width: 360px;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    .meta {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }}
    .grade-panel {{
      border-left: 1px solid var(--border);
      padding-left: 14px;
    }}
    .grade-buttons {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin: 8px 0 10px;
    }}
    .grade {{
      font-weight: 700;
      padding: 10px 0;
    }}
    .grade.selected[data-grade="0"] {{
      background: #ffebe9;
      border-color: #cf222e;
    }}
    .grade.selected[data-grade="1"] {{
      background: #fff8c5;
      border-color: #9a6700;
    }}
    .grade.selected[data-grade="2"] {{
      background: #dafbe1;
      border-color: #1a7f37;
    }}
    textarea {{
      width: 100%;
      min-height: 70px;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px;
      resize: vertical;
      font: inherit;
    }}
    .rubric {{
      font-size: 13px;
      line-height: 1.45;
      color: var(--muted);
      margin-top: 8px;
    }}
    .hidden {{
      display: none;
    }}
    @media (max-width: 900px) {{
      .pair {{
        grid-template-columns: 1fr;
      }}
      .grade-panel {{
        border-left: 0;
        border-top: 1px solid var(--border);
        padding-left: 0;
        padding-top: 12px;
      }}
      .stats {{
        margin-left: 0;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Manual Relevance Annotation</h1>
    <div class="toolbar">
      <button type="button" class="primary" onclick="downloadCsv()">Download annotated CSV</button>
      <button type="button" onclick="saveState()">Save in browser</button>
      <button type="button" onclick="clearFilters()">Show all</button>
      <select id="filter" onchange="render()">
        <option value="all">All rows</option>
        <option value="ungraded">Ungraded only</option>
        <option value="graded">Graded only</option>
      </select>
      <input id="queryFilter" placeholder="Filter query_id..." oninput="render()">
      <span class="stats" id="stats"></span>
    </div>
  </header>
  <main>
    <p class="rubric">
      Grade each candidate image against the query image by visual similarity, not by species correctness.
      <strong>0</strong> = irrelevant, <strong>1</strong> = partially similar,
      <strong>2</strong> = highly similar. Consider color, texture, shape, pose, and regional layout.
    </p>
    <div id="cards"></div>
  </main>
  <script>
    const rows = {rows_json};
    const storageKey = "manual_relevance_annotation_v1";
    const columns = [
      "query_id",
      "query_image_id",
      "query_species_name",
      "query_processed_relative_path",
      "candidate_image_id",
      "candidate_species_name",
      "candidate_processed_relative_path",
      "source_experiments",
      "best_rank_seen",
      "relevance_grade",
      "review_notes"
    ];

    function loadState() {{
      const saved = localStorage.getItem(storageKey);
      if (!saved) return;
      const state = JSON.parse(saved);
      for (const row of rows) {{
        const key = rowKey(row);
        if (state[key]) {{
          row.relevance_grade = state[key].relevance_grade || row.relevance_grade || "";
          row.review_notes = state[key].review_notes || row.review_notes || "";
        }}
      }}
    }}

    function rowKey(row) {{
      return `${{row.query_id}}::${{row.candidate_image_id}}`;
    }}

    function saveState() {{
      const state = {{}};
      for (const row of rows) {{
        state[rowKey(row)] = {{
          relevance_grade: row.relevance_grade || "",
          review_notes: row.review_notes || ""
        }};
      }}
      localStorage.setItem(storageKey, JSON.stringify(state));
      updateStats();
    }}

    function setGrade(index, grade) {{
      rows[index].relevance_grade = grade;
      saveState();
      render();
    }}

    function setNotes(index, value) {{
      rows[index].review_notes = value;
      saveState();
    }}

    function filteredRows() {{
      const filter = document.getElementById("filter").value;
      const queryFilter = document.getElementById("queryFilter").value.trim();
      return rows
        .map((row, index) => [row, index])
        .filter(([row]) => {{
          const graded = String(row.relevance_grade || "").trim() !== "";
          if (filter === "ungraded" && graded) return false;
          if (filter === "graded" && !graded) return false;
          if (queryFilter && String(row.query_id) !== queryFilter) return false;
          return true;
        }});
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function render() {{
      const root = document.getElementById("cards");
      const items = filteredRows();
      root.innerHTML = items.map(([row, index]) => cardHtml(row, index)).join("");
      updateStats(items.length);
    }}

    function cardHtml(row, index) {{
      const grade = String(row.relevance_grade || "");
      return `
        <section class="card">
          <div class="card-head">
            <span><strong>Row:</strong> ${{row.row_index}}</span>
            <span><strong>query_id:</strong> ${{escapeHtml(row.query_id)}}</span>
            <span><strong>best rank seen:</strong> ${{escapeHtml(row.best_rank_seen)}}</span>
            <span><strong>source experiments:</strong> ${{escapeHtml(row.source_experiments)}}</span>
          </div>
          <div class="pair">
            <div class="image-block">
              <h3>Query image</h3>
              <img src="${{escapeHtml(row.query_img_src)}}" alt="query image">
              <div class="meta">
                image_id: ${{escapeHtml(row.query_image_id)}}<br>
                ${{escapeHtml(row.query_species_name)}}
              </div>
            </div>
            <div class="image-block">
              <h3>Candidate image</h3>
              <img src="${{escapeHtml(row.candidate_img_src)}}" alt="candidate image">
              <div class="meta">
                image_id: ${{escapeHtml(row.candidate_image_id)}}<br>
                ${{escapeHtml(row.candidate_species_name)}}
              </div>
            </div>
            <div class="grade-panel">
              <strong>Relevance grade</strong>
              <div class="grade-buttons">
                ${{["0", "1", "2"].map(g => `
                  <button type="button"
                    class="grade ${{grade === g ? "selected" : ""}}"
                    data-grade="${{g}}"
                    onclick="setGrade(${{index}}, '${{g}}')">${{g}}</button>
                `).join("")}}
              </div>
              <textarea placeholder="Optional notes..." oninput="setNotes(${{index}}, this.value)">${{escapeHtml(row.review_notes || "")}}</textarea>
              <div class="rubric">
                0: visually different<br>
                1: shares some color/pose/texture/shape<br>
                2: strongly similar in multiple visual aspects
              </div>
            </div>
          </div>
        </section>
      `;
    }}

    function updateStats(visibleCount = filteredRows().length) {{
      const graded = rows.filter(row => String(row.relevance_grade || "").trim() !== "").length;
      document.getElementById("stats").textContent =
        `${{graded}} / ${{rows.length}} graded | ${{visibleCount}} visible`;
    }}

    function clearFilters() {{
      document.getElementById("filter").value = "all";
      document.getElementById("queryFilter").value = "";
      render();
    }}

    function csvEscape(value) {{
      const text = String(value ?? "");
      if (/[",\\n\\r]/.test(text)) {{
        return '"' + text.replaceAll('"', '""') + '"';
      }}
      return text;
    }}

    function downloadCsv() {{
      saveState();
      const lines = [columns.join(",")];
      for (const row of rows) {{
        lines.push(columns.map(col => csvEscape(row[col] || "")).join(","));
      }}
      const blob = new Blob([lines.join("\\n") + "\\n"], {{type: "text/csv;charset=utf-8"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "manual_relevance_annotated.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    loadState();
    render();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    template_csv = Path(args.template_csv).resolve()
    processed_root = Path(args.processed_root).resolve()
    output_html = Path(args.output_html).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(template_csv, output_html, processed_root)
    if not rows:
        raise ValueError(f"No rows found in {template_csv}")

    output_html.write_text(build_html(rows), encoding="utf-8")
    print(f"Saved manual relevance gallery: {output_html}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
