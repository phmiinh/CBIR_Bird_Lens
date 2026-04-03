from __future__ import annotations

import argparse
import csv
import collections
import html
import json
import math
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

from cub_utils import (
    draw_bbox,
    load_cub_records,
    load_image_attribute_presence,
    load_part_visibility,
    normalize_image,
    write_csv,
)

PERCHING_ATTRIBUTE_ID = 236
UPRIGHT_PERCHING_WATER_ATTRIBUTE_ID = 223
TREE_CLINGING_ATTRIBUTE_ID = 231

LEFT_EYE_PART_ID = 7
LEFT_LEG_PART_ID = 8
LEFT_WING_PART_ID = 9
RIGHT_EYE_PART_ID = 11
RIGHT_LEG_PART_ID = 12
RIGHT_WING_PART_ID = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample CUB images and generate a review gallery for manual filtering."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/raw/cub2002011",
        help="Directory containing the extracted CUB dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of candidates to include in the review set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.20,
        help="Extra padding added around the bounding box before square crop preview.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/review",
        help="Directory that will receive the CSV manifest and HTML gallery.",
    )
    parser.add_argument(
        "--reuse-selection-csv",
        default="",
        help="Reuse image list/order from an existing candidates or reviewed CSV instead of re-sampling.",
    )
    parser.add_argument(
        "--prefill-keep-csv",
        default="",
        help="Prefill keep/notes from a reviewed CSV (useful when regenerating previews with a new padding ratio).",
    )
    parser.add_argument(
        "--exclude-reviewed-csv",
        action="append",
        default=[],
        help="Exclude all image_id entries that already appear in a previous candidates/reviewed CSV. Repeat this flag to exclude multiple earlier batches.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("assignment", "random"),
        default="assignment",
        help="Use heuristic filtering for perching/side-view assignment images or plain random sampling.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=2,
        help="Sample from the best N*sample_size candidates when using assignment mode.",
    )
    parser.add_argument(
        "--distribution-mode",
        choices=("species_balanced", "ranked"),
        default="species_balanced",
        help="How to distribute candidates inside the sampled set.",
    )
    return parser.parse_args()


def build_preview(source_image: Image.Image, normalized_image: Image.Image, preview_path: Path) -> None:
    left = source_image.copy()
    left.thumbnail((360, 360))
    right = normalized_image.copy()
    right.thumbnail((224, 224))

    canvas_width = left.width + right.width + 24
    canvas_height = max(left.height, right.height) + 16
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(248, 248, 248))
    canvas.paste(left, (8, 8))
    canvas.paste(right, (left.width + 16, 8))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path, quality=90)


def render_gallery(cards: List[dict], output_path: Path) -> None:
    gallery_rows = []
    for card in cards:
        row = dict(card)
        row["preview_src"] = f'previews/{card["preview_file"]}'
        row["title"] = f'Image ID {card["image_id"]} - {card["species_name"]}'
        row["subtitle"] = card["relative_image_path"]
        gallery_rows.append(row)

    cards_json = json.dumps(gallery_rows, ensure_ascii=True)
    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUB Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4efe6;
      --ink: #1f1a16;
      --muted: #6a5f53;
      --card: rgba(255, 251, 245, 0.94);
      --line: rgba(113, 86, 61, 0.18);
      --accent: #8f3f1f;
      --accent-strong: #6d2d14;
      --accent-soft: #f3dfcf;
      --keep: #145a32;
      --keep-soft: #dff3e6;
      --skip: #7b241c;
      --skip-soft: #f8e0dc;
      --shadow: 0 18px 40px rgba(65, 42, 20, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(143, 63, 31, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(195, 152, 116, 0.18), transparent 26%),
        linear-gradient(180deg, #fbf6ef 0%, var(--bg) 100%);
    }
    .shell {
      width: min(1420px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }
    .hero {
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 20px 24px 18px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 250, 243, 0.88);
      backdrop-filter: blur(14px);
      box-shadow: var(--shadow);
    }
    .hero-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }
    .title-block h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1;
      letter-spacing: -0.03em;
      color: var(--accent-strong);
    }
    .title-block p {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
    }
    .summary {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .stat {
      min-width: 124px;
      padding: 10px 14px;
      border-radius: 18px;
      background: white;
      border: 1px solid var(--line);
      box-shadow: 0 10px 24px rgba(65, 42, 20, 0.06);
    }
    .stat strong {
      display: block;
      font-size: 22px;
      color: var(--accent-strong);
    }
    .stat span {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .pager {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .filters {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .filters label {
      color: var(--muted);
      font-size: 14px;
    }
    .species-select {
      min-width: 320px;
      max-width: min(42vw, 520px);
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      padding: 8px 12px;
      font: inherit;
    }
    .nav-btn,
    .page-btn,
    .action-btn {
      appearance: none;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }
    .nav-btn:hover,
    .page-btn:hover,
    .action-btn:hover {
      transform: translateY(-1px);
      border-color: rgba(143, 63, 31, 0.35);
    }
    .nav-btn[disabled] {
      opacity: 0.4;
      cursor: not-allowed;
      transform: none;
    }
    .page-list {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .page-btn.active {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .page-gap {
      color: var(--muted);
      padding: 0 4px;
    }
    .page-info {
      color: var(--muted);
      font-size: 15px;
      margin-left: 6px;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .action-btn.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .action-btn.ghost {
      background: transparent;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
      margin-top: 20px;
      align-items: start;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 14px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .card.is-keep {
      border-color: rgba(20, 90, 50, 0.35);
      box-shadow: 0 18px 40px rgba(20, 90, 50, 0.12);
    }
    .card.is-skip {
      border-color: rgba(123, 36, 28, 0.24);
      box-shadow: 0 18px 40px rgba(123, 36, 28, 0.08);
    }
    .preview-trigger {
      display: block;
      width: 100%;
      padding: 0;
      border: 0;
      background: transparent;
      cursor: zoom-in;
    }
    .card img {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #f1e6da;
      object-fit: contain;
    }
    .card-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin: 12px 0 10px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 70px;
      padding: 5px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
    }
    .status {
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border: 1px solid transparent;
      color: var(--muted);
      background: rgba(106, 95, 83, 0.08);
    }
    .status.keep {
      color: var(--keep);
      background: var(--keep-soft);
      border-color: rgba(20, 90, 50, 0.16);
    }
    .status.skip {
      color: var(--skip);
      background: var(--skip-soft);
      border-color: rgba(123, 36, 28, 0.16);
    }
    .card h3 {
      margin: 0;
      font-size: 19px;
      line-height: 1.25;
      color: var(--accent-strong);
    }
    .card-actions {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      margin-top: 14px;
    }
    .pick-btn {
      appearance: none;
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      font: inherit;
      cursor: pointer;
      background: white;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }
    .pick-btn:hover {
      transform: translateY(-1px);
    }
    .pick-btn.keep.active {
      background: var(--keep);
      border-color: var(--keep);
      color: white;
    }
    .pick-btn.skip.active {
      background: var(--skip);
      border-color: var(--skip);
      color: white;
    }
    .pick-btn.clear.active {
      background: var(--accent-soft);
      border-color: rgba(143, 63, 31, 0.20);
      color: var(--accent-strong);
    }
    .footer-note {
      margin-top: 18px;
      color: var(--muted);
      font-size: 14px;
    }
    .modal[hidden] {
      display: none;
    }
    .modal {
      position: fixed;
      inset: 0;
      z-index: 30;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(24, 18, 14, 0.78);
      backdrop-filter: blur(10px);
    }
    .modal-panel {
      width: min(96vw, 1560px);
      max-height: 92vh;
      padding: 16px;
      border-radius: 24px;
      background: rgba(255, 251, 245, 0.98);
      border: 1px solid rgba(255, 255, 255, 0.35);
      box-shadow: 0 26px 80px rgba(0, 0, 0, 0.32);
    }
    .modal-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }
    .modal-title {
      margin: 0;
      font-size: 18px;
      color: var(--accent-strong);
    }
    .modal-close {
      appearance: none;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 14px;
      font: inherit;
      cursor: pointer;
    }
    .modal-image-wrap {
      max-height: calc(92vh - 92px);
      overflow: auto;
      border-radius: 18px;
      background: #efe3d4;
      border: 1px solid var(--line);
    }
    .modal-image {
      display: block;
      width: 100%;
      height: auto;
    }
    @media (max-width: 860px) {
      .shell {
        width: min(100% - 18px, 100%);
      }
      .hero {
        padding: 18px 16px 16px;
        border-radius: 20px;
      }
      .hero-top,
      .controls {
        flex-direction: column;
        align-items: stretch;
      }
      .summary {
        justify-content: flex-start;
      }
      .card-actions {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 720px) {
      .gallery {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-top">
        <div class="title-block">
          <h1>Bird Review Board</h1>
          <p>One species per page. Use Keep for final candidates, then export a reviewed CSV.</p>
        </div>
        <div class="summary">
          <div class="stat"><strong id="selectedCount">0</strong><span>Selected</span></div>
          <div class="stat"><strong id="skippedCount">0</strong><span>Skipped</span></div>
          <div class="stat"><strong id="remainingCount">0</strong><span>Unreviewed</span></div>
        </div>
      </div>
      <div class="controls">
        <div class="pager">
          <button id="firstBtn" class="nav-btn" type="button">First</button>
          <button id="prevBtn" class="nav-btn" type="button">Previous</button>
          <div id="pageList" class="page-list" aria-label="Pages"></div>
          <button id="nextBtn" class="nav-btn" type="button">Next</button>
          <button id="lastBtn" class="nav-btn" type="button">Last</button>
          <span id="pageInfo" class="page-info"></span>
        </div>
        <div class="filters">
          <label for="speciesSelect">Jump To Species</label>
          <select id="speciesSelect" class="species-select"></select>
        </div>
        <div class="actions">
          <button id="downloadBtn" class="action-btn primary" type="button">Download Reviewed CSV</button>
          <button id="resetBtn" class="action-btn ghost" type="button">Clear Local Review</button>
        </div>
      </div>
    </section>
    <main id="gallery" class="gallery"></main>
    <div id="previewModal" class="modal" hidden>
      <div class="modal-panel">
        <div class="modal-bar">
          <h2 id="modalTitle" class="modal-title"></h2>
          <button id="closeModalBtn" class="modal-close" type="button">Close</button>
        </div>
        <div class="modal-image-wrap">
          <img id="modalImage" class="modal-image" alt="">
        </div>
      </div>
    </div>
    <p class="footer-note">The HTML keeps your review state in browser local storage for this file path.</p>
  </div>

  <script>
    const rows = __CARDS_JSON__;
    const storageKey = 'cub-review:' + window.location.pathname + ':' + rows.length;
    const gallery = document.getElementById('gallery');
    const previewModal = document.getElementById('previewModal');
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('modalTitle');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const pageList = document.getElementById('pageList');
    const pageInfo = document.getElementById('pageInfo');
    const speciesSelect = document.getElementById('speciesSelect');
    const selectedCount = document.getElementById('selectedCount');
    const skippedCount = document.getElementById('skippedCount');
    const remainingCount = document.getElementById('remainingCount');
    const firstBtn = document.getElementById('firstBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const lastBtn = document.getElementById('lastBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const resetBtn = document.getElementById('resetBtn');

    function buildSpeciesGroups(sourceRows) {
      const buckets = new Map();
      sourceRows.forEach((row) => {
        const speciesId = Number(row.species_id);
        const key = String(speciesId) + '|' + row.species_name;
        if (!buckets.has(key)) {
          buckets.set(key, {
            species_id: speciesId,
            species_name: row.species_name,
            rows: [],
          });
        }
        buckets.get(key).rows.push(row);
      });

      const groups = Array.from(buckets.values()).sort((left, right) => left.species_id - right.species_id);
      groups.forEach((group) => {
        group.rows.sort((left, right) => Number(left.review_index) - Number(right.review_index));
      });
      return groups;
    }

    const speciesGroups = buildSpeciesGroups(rows);
    let currentPage = 1;

    function loadReviewState() {
      try {
        const raw = localStorage.getItem(storageKey);
        return raw ? JSON.parse(raw) : {};
      } catch (_error) {
        return {};
      }
    }

    function saveReviewState() {
      const state = {};
      rows.forEach((row) => {
        if (row.review_status) {
          state[row.image_id] = row.review_status;
        }
      });
      localStorage.setItem(storageKey, JSON.stringify(state));
    }

    function initializeRows() {
      const persisted = loadReviewState();
      rows.forEach((row) => {
        if (!('review_status' in row)) {
          row.review_status = '';
        }
        if (persisted[row.image_id]) {
          row.review_status = persisted[row.image_id];
        } else if (String(row.keep || '').trim() === '1') {
          row.review_status = 'keep';
        } else if (String(row.keep || '').trim() === '0') {
          row.review_status = 'skip';
        }
      });
    }

    function totalPages() {
      return Math.max(1, speciesGroups.length);
    }

    function currentGroup() {
      return speciesGroups[currentPage - 1] || { species_id: 0, species_name: 'N/A', rows: [] };
    }

    function populateSpeciesFilter() {
      if (speciesGroups.length === 0) {
        speciesSelect.innerHTML = '<option value="1">No species</option>';
        return;
      }

      const options = [];
      speciesGroups.forEach((group, index) => {
        const safeName = escapeHtml(group.species_name);
        const pageNo = index + 1;
        options.push(`<option value="${pageNo}">${String(pageNo).padStart(3, '0')} - ${safeName} (${group.rows.length})</option>`);
      });
      speciesSelect.innerHTML = options.join('');
      speciesSelect.value = String(currentPage);
    }

    function pageSequence() {
      const total = totalPages();
      const pages = new Set([1, total, currentPage - 1, currentPage, currentPage + 1]);
      return Array.from(pages)
        .filter((page) => page >= 1 && page <= total)
        .sort((left, right) => left - right);
    }

    function summarize() {
      const selected = rows.filter((row) => row.review_status === 'keep').length;
      const skipped = rows.filter((row) => row.review_status === 'skip').length;
      const remaining = rows.length - selected - skipped;
      selectedCount.textContent = String(selected);
      skippedCount.textContent = String(skipped);
      remainingCount.textContent = String(remaining);
    }

    function statusMarkup(status) {
      if (status === 'keep') {
        return '<span class="status keep">Keep</span>';
      }
      if (status === 'skip') {
        return '<span class="status skip">Skip</span>';
      }
      return '<span class="status">Pending</span>';
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function renderCards() {
      const pageRows = currentGroup().rows;
      gallery.innerHTML = pageRows.map((row) => {
        const keepActive = row.review_status === 'keep' ? ' active' : '';
        const skipActive = row.review_status === 'skip' ? ' active' : '';
        const clearActive = row.review_status === '' ? ' active' : '';
        const cardState = row.review_status ? ' is-' + row.review_status : '';
        return `
          <article class="card${cardState}" data-image-id="${row.image_id}">
            <button class="preview-trigger" type="button" data-preview-src="${escapeHtml(row.preview_src)}" data-preview-title="${escapeHtml(row.title)}">
              <img src="${escapeHtml(row.preview_src)}" alt="${escapeHtml(row.title)}">
            </button>
            <div class="card-head">
              <span class="badge">#${String(row.review_index).padStart(4, '0')}</span>
              ${statusMarkup(row.review_status)}
            </div>
            <h3>${escapeHtml(row.title)}</h3>
            <div class="card-actions">
              <button class="pick-btn keep${keepActive}" type="button" data-action="keep" data-image-id="${row.image_id}">Keep</button>
              <button class="pick-btn skip${skipActive}" type="button" data-action="skip" data-image-id="${row.image_id}">Skip</button>
              <button class="pick-btn clear${clearActive}" type="button" data-action="clear" data-image-id="${row.image_id}">Reset</button>
            </div>
          </article>
        `;
      }).join('');
    }

    function renderPagination() {
      const group = currentGroup();
      pageList.innerHTML = '';
      let previousPage = 0;
      pageSequence().forEach((page) => {
        if (previousPage && page - previousPage > 1) {
          const gap = document.createElement('span');
          gap.className = 'page-gap';
          gap.textContent = '...';
          pageList.appendChild(gap);
        }

        const button = document.createElement('button');
        button.type = 'button';
        button.className = page === currentPage ? 'page-btn active' : 'page-btn';
        button.textContent = String(page);
        button.addEventListener('click', () => goToPage(page, true));
        pageList.appendChild(button);
        previousPage = page;
      });

      const total = totalPages();
      const speciesLabel = String(group.species_id).padStart(3, '0') + '.' + group.species_name;
      pageInfo.textContent = 'Species Page ' + currentPage + '/' + total + ' | ' + speciesLabel + ' | ' + group.rows.length + ' images';
      speciesSelect.value = String(currentPage);
      firstBtn.disabled = currentPage === 1;
      prevBtn.disabled = currentPage === 1;
      nextBtn.disabled = currentPage === total;
      lastBtn.disabled = currentPage === total;
    }

    function render() {
      if (currentPage > totalPages()) {
        currentPage = totalPages();
      }
      summarize();
      renderPagination();
      renderCards();
    }

    function goToPage(page, scrollToTop = false) {
      currentPage = Math.max(1, Math.min(page, totalPages()));
      render();
      if (scrollToTop) {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }

    function setReviewStatus(imageId, status) {
      const row = rows.find((item) => String(item.image_id) === String(imageId));
      if (!row) {
        return;
      }
      row.review_status = status;
      saveReviewState();
      render();
    }

    function openPreview(src, title) {
      modalImage.src = src;
      modalImage.alt = title;
      modalTitle.textContent = title;
      previewModal.hidden = false;
      document.body.style.overflow = 'hidden';
    }

    function closePreview() {
      previewModal.hidden = true;
      modalImage.src = '';
      modalImage.alt = '';
      modalTitle.textContent = '';
      document.body.style.overflow = '';
    }

    function buildReviewedCsv() {
      const header = Object.keys(rows[0]);
      const exportHeader = header.filter((field) => !['preview_src', 'title', 'subtitle', 'review_status'].includes(field));
      const csvRows = [exportHeader.join(',')];

      rows.forEach((row) => {
        const exportRow = { ...row };
        if (row.review_status === 'keep') {
          exportRow.keep = '1';
        } else if (row.review_status === 'skip') {
          exportRow.keep = '0';
        } else {
          exportRow.keep = '';
        }

        const line = exportHeader.map((field) => {
          const value = exportRow[field] == null ? '' : String(exportRow[field]);
          const escaped = value.replaceAll('"', '""');
          return '"' + escaped + '"';
        }).join(',');
        csvRows.push(line);
      });

      return csvRows.join('\\r\\n');
    }

    function downloadReviewedCsv() {
      const csvText = buildReviewedCsv();
      const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'reviewed_candidates.csv';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    gallery.addEventListener('click', (event) => {
      const previewTrigger = event.target.closest('.preview-trigger');
      if (previewTrigger) {
        openPreview(previewTrigger.dataset.previewSrc, previewTrigger.dataset.previewTitle);
        return;
      }

      const target = event.target.closest('button[data-action]');
      if (!target) {
        return;
      }

      const action = target.dataset.action;
      const imageId = target.dataset.imageId;
      if (action === 'keep') {
        setReviewStatus(imageId, 'keep');
      } else if (action === 'skip') {
        setReviewStatus(imageId, 'skip');
      } else {
        setReviewStatus(imageId, '');
      }
    });

    firstBtn.addEventListener('click', () => goToPage(1, true));
    prevBtn.addEventListener('click', () => goToPage(currentPage - 1, true));
    nextBtn.addEventListener('click', () => goToPage(currentPage + 1, true));
    lastBtn.addEventListener('click', () => goToPage(totalPages(), true));
    speciesSelect.addEventListener('change', () => {
      const targetPage = Number(speciesSelect.value);
      if (Number.isFinite(targetPage)) {
        goToPage(targetPage, false);
      }
    });
    downloadBtn.addEventListener('click', downloadReviewedCsv);
    closeModalBtn.addEventListener('click', closePreview);
    previewModal.addEventListener('click', (event) => {
      if (event.target === previewModal) {
        closePreview();
      }
    });
    window.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !previewModal.hidden) {
        closePreview();
      }
    });
    resetBtn.addEventListener('click', () => {
      if (!window.confirm('Clear all local Keep/Skip decisions for this gallery?')) {
        return;
      }
      rows.forEach((row) => {
        row.review_status = '';
      });
      localStorage.removeItem(storageKey);
      render();
    });

    initializeRows();
    populateSpeciesFilter();
    render();
  </script>
</body>
</html>
"""

    document = document.replace("__CARDS_JSON__", cards_json)
    output_path.write_text(document, encoding="utf-8")


def enrich_records(records: List[dict], dataset_root: Path) -> List[dict]:
    attribute_presence = load_image_attribute_presence(
        dataset_root,
        attribute_ids=(
            PERCHING_ATTRIBUTE_ID,
            UPRIGHT_PERCHING_WATER_ATTRIBUTE_ID,
            TREE_CLINGING_ATTRIBUTE_ID,
        ),
    )
    part_visibility = load_part_visibility(
        dataset_root,
        part_ids=(
            LEFT_EYE_PART_ID,
            RIGHT_EYE_PART_ID,
            LEFT_WING_PART_ID,
            RIGHT_WING_PART_ID,
            LEFT_LEG_PART_ID,
            RIGHT_LEG_PART_ID,
        ),
    )

    enriched = []
    for record in records:
        image_id = record["image_id"]
        attrs = attribute_presence.get(image_id, {})
        parts = part_visibility.get(image_id, {})

        perching_like = int(attrs.get(PERCHING_ATTRIBUTE_ID, 0) == 1)
        upright_water_like = int(attrs.get(UPRIGHT_PERCHING_WATER_ATTRIBUTE_ID, 0) == 1)
        tree_clinging_like = int(attrs.get(TREE_CLINGING_ATTRIBUTE_ID, 0) == 1)

        visible_eyes = int(parts.get(LEFT_EYE_PART_ID, 0)) + int(parts.get(RIGHT_EYE_PART_ID, 0))
        visible_wings = int(parts.get(LEFT_WING_PART_ID, 0)) + int(parts.get(RIGHT_WING_PART_ID, 0))
        visible_legs = int(parts.get(LEFT_LEG_PART_ID, 0)) + int(parts.get(RIGHT_LEG_PART_ID, 0))

        one_eye_visible = int(visible_eyes == 1)
        one_wing_visible = int(visible_wings == 1)
        legs_visible = int(visible_legs >= 1)
        bbox_horizontal = int(record["bbox_w"] >= record["bbox_h"])

        assignment_score = (
            (5 * perching_like)
            + (2 * one_eye_visible)
            + (2 * one_wing_visible)
            + (1 * legs_visible)
            + (1 * bbox_horizontal)
            - (2 * upright_water_like)
            - (3 * tree_clinging_like)
        )

        enriched_record = dict(record)
        enriched_record.update(
            {
                "perching_like": perching_like,
                "upright_water_like": upright_water_like,
                "tree_clinging_like": tree_clinging_like,
                "one_eye_visible": one_eye_visible,
                "one_wing_visible": one_wing_visible,
                "legs_visible": legs_visible,
                "bbox_horizontal": bbox_horizontal,
                "assignment_score": assignment_score,
            }
        )
        enriched.append(enriched_record)

    return enriched


def _build_base_pool(
    records: List[dict],
    sample_size: int,
    seed: int,
    sampling_strategy: str,
    candidate_multiplier: int,
    distribution_mode: str,
) -> List[dict]:
    rng = random.Random(seed)

    if sampling_strategy == "random":
        pool = list(records)
        rng.shuffle(pool)
        return pool

    if distribution_mode == "species_balanced":
        by_species: Dict[int, List[dict]] = {}
        for record in records:
            by_species.setdefault(int(record["species_id"]), []).append(record)

        species_count = max(1, len(by_species))
        per_species_cap = max(1, math.ceil((sample_size * max(1, candidate_multiplier)) / species_count))
        pool: List[dict] = []
        for species_id in sorted(by_species):
            bucket = by_species[species_id]
            bucket.sort(key=lambda item: (-item["assignment_score"], item["image_id"]))
            pool.extend(bucket[:per_species_cap])
        return pool

    perching_candidates = [record for record in records if record["perching_like"] == 1]
    if len(perching_candidates) < sample_size:
        perching_candidates = list(records)

    ranked_candidates = sorted(
        perching_candidates,
        key=lambda item: (-item["assignment_score"], item["image_id"]),
    )
    top_pool_size = min(len(ranked_candidates), max(sample_size, sample_size * max(1, candidate_multiplier)))
    return ranked_candidates[:top_pool_size]


def _species_balanced_sample(pool: List[dict], sample_size: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    by_species: Dict[int, List[dict]] = {}
    for record in pool:
        by_species.setdefault(int(record["species_id"]), []).append(record)

    for species_id, records in by_species.items():
        records.sort(key=lambda item: (-item["assignment_score"], item["image_id"]))
        by_species[species_id] = records

    species_ids = sorted(by_species.keys())
    if not species_ids:
        return []

    rng.shuffle(species_ids)
    base_quota = sample_size // len(species_ids)
    remainder = sample_size % len(species_ids)

    selected: List[dict] = []
    indices: Dict[int, int] = {}
    for species_index, species_id in enumerate(species_ids):
        bucket = by_species[species_id]
        quota = base_quota + (1 if species_index < remainder else 0)
        take_count = min(quota, len(bucket))
        selected.extend(bucket[:take_count])
        indices[species_id] = take_count

    missing = sample_size - len(selected)
    active_species = [species_id for species_id in species_ids if indices[species_id] < len(by_species[species_id])]

    while missing > 0 and active_species:
        rng.shuffle(active_species)
        next_active: List[int] = []
        for species_id in active_species:
            if missing <= 0:
                break
            next_index = indices[species_id]
            bucket = by_species[species_id]
            if next_index < len(bucket):
                selected.append(bucket[next_index])
                indices[species_id] = next_index + 1
                missing -= 1
            if indices[species_id] < len(bucket):
                next_active.append(species_id)
        active_species = next_active

    selected.sort(key=lambda item: (item["species_id"], -item["assignment_score"], item["image_id"]))
    return selected[:sample_size]


def choose_records(
    records: List[dict],
    sample_size: int,
    seed: int,
    sampling_strategy: str,
    candidate_multiplier: int,
    distribution_mode: str,
) -> List[dict]:
    base_pool = _build_base_pool(
        records,
        sample_size=sample_size,
        seed=seed,
        sampling_strategy=sampling_strategy,
        candidate_multiplier=candidate_multiplier,
        distribution_mode=distribution_mode,
    )

    if sample_size > len(base_pool):
        raise ValueError(
            f"Sample size {sample_size} cannot be satisfied from candidate pool size {len(base_pool)}."
        )

    rng = random.Random(seed)
    if distribution_mode == "ranked":
        chosen = rng.sample(base_pool, sample_size)
        chosen.sort(key=lambda item: (-item["assignment_score"], item["image_id"]))
        return chosen

    return _species_balanced_sample(base_pool, sample_size=sample_size, seed=seed)


def _read_csv_rows(file_path: Path) -> List[dict]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_image_id(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _normalize_keep_value(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    keep_values = {"1", "true", "yes", "y", "keep", "x", "selected", "ok"}
    skip_values = {"0", "false", "no", "n", "skip", "reject", "rejected"}
    if text in keep_values:
        return "1"
    if text in skip_values:
        return "0"
    return ""


def load_prefill_state(file_path: Path) -> Dict[int, Dict[str, str]]:
    rows = _read_csv_rows(file_path)
    state: Dict[int, Dict[str, str]] = {}
    for row in rows:
        image_id = _parse_image_id(row.get("image_id", ""))
        if image_id is None:
            continue
        state[image_id] = {
            "keep": _normalize_keep_value(row.get("keep", "")),
            "notes": str(row.get("notes", "") or "").strip(),
        }
    return state


def load_excluded_image_ids(file_paths: List[Path]) -> set[int]:
    excluded: set[int] = set()
    for file_path in file_paths:
        rows = _read_csv_rows(file_path)
        for row in rows:
            image_id = _parse_image_id(row.get("image_id", ""))
            if image_id is not None:
                excluded.add(image_id)
    return excluded


def reuse_sampled_records(selection_csv: Path, enriched_records: List[dict]) -> List[dict]:
    rows = _read_csv_rows(selection_csv)
    enriched_by_id = {int(record["image_id"]): record for record in enriched_records}

    ordered_items = []
    for fallback_index, row in enumerate(rows, start=1):
        image_id = _parse_image_id(row.get("image_id", ""))
        if image_id is None:
            continue
        review_index = _parse_image_id(row.get("review_index", "")) or fallback_index
        ordered_items.append((review_index, fallback_index, image_id))

    ordered_items.sort(key=lambda item: (item[0], item[1]))
    sampled_records: List[dict] = []
    seen_ids = set()
    missing_ids = []
    for _, _, image_id in ordered_items:
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)
        record = enriched_by_id.get(image_id)
        if record is None:
            missing_ids.append(image_id)
            continue
        sampled_records.append(record)

    if not sampled_records:
        raise ValueError(f"No reusable image rows were found in {selection_csv}.")
    if missing_ids:
        print(f"Warning: {len(missing_ids)} image_id entries from {selection_csv} were not found in dataset metadata.")
    return sampled_records


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    records = load_cub_records(dataset_root)
    if args.sample_size > len(records):
        raise ValueError(f"Sample size {args.sample_size} is larger than dataset size {len(records)}.")

    enriched_records = enrich_records(records, dataset_root)
    reuse_selection_csv = Path(args.reuse_selection_csv).resolve() if args.reuse_selection_csv else None
    prefill_keep_csv = Path(args.prefill_keep_csv).resolve() if args.prefill_keep_csv else None
    exclude_reviewed_csvs = [Path(value).resolve() for value in args.exclude_reviewed_csv]

    if reuse_selection_csv:
        if not reuse_selection_csv.exists():
            raise FileNotFoundError(f"Could not find reuse-selection CSV: {reuse_selection_csv}")
        sampled_records = reuse_sampled_records(reuse_selection_csv, enriched_records)
        print(f"Reused candidate list from: {reuse_selection_csv} ({len(sampled_records)} images)")
        if prefill_keep_csv is None:
            prefill_keep_csv = reuse_selection_csv
    else:
        for file_path in exclude_reviewed_csvs:
            if not file_path.exists():
                raise FileNotFoundError(f"Could not find exclude-reviewed CSV: {file_path}")
        if exclude_reviewed_csvs:
            excluded_image_ids = load_excluded_image_ids(exclude_reviewed_csvs)
            enriched_records = [
                record for record in enriched_records if int(record["image_id"]) not in excluded_image_ids
            ]
            print(
                f"Excluded {len(excluded_image_ids)} previously reviewed image_ids from "
                f"{len(exclude_reviewed_csvs)} CSV file(s)."
            )
            if args.sample_size > len(enriched_records):
                raise ValueError(
                    f"Sample size {args.sample_size} cannot be satisfied after exclusions. "
                    f"Remaining candidates: {len(enriched_records)}."
                )
        sampled_records = choose_records(
            enriched_records,
            sample_size=args.sample_size,
            seed=args.seed,
            sampling_strategy=args.sampling_strategy,
            candidate_multiplier=args.candidate_multiplier,
            distribution_mode=args.distribution_mode,
        )

    species_counter = collections.Counter(record["species_id"] for record in sampled_records)
    min_per_species = min(species_counter.values()) if species_counter else 0
    max_per_species = max(species_counter.values()) if species_counter else 0
    print(
        f"Species coverage: {len(species_counter)} species | "
        f"min per species: {min_per_species} | max per species: {max_per_species}"
    )
    prefill_state: Dict[int, Dict[str, str]] = {}
    if prefill_keep_csv:
        if not prefill_keep_csv.exists():
            raise FileNotFoundError(f"Could not find prefill-keep CSV: {prefill_keep_csv}")
        prefill_state = load_prefill_state(prefill_keep_csv)
        print(f"Prefilled keep/notes from: {prefill_keep_csv} ({len(prefill_state)} rows)")

    csv_rows = []
    gallery_cards = []
    for review_index, record in enumerate(tqdm(sampled_records, desc="Generating previews"), start=1):
        image_path = Path(record["absolute_image_path"])
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            annotated = draw_bbox(
                rgb_image,
                (record["bbox_x"], record["bbox_y"], record["bbox_w"], record["bbox_h"]),
            )
            normalized, crop_box = normalize_image(
                rgb_image,
                (record["bbox_x"], record["bbox_y"], record["bbox_w"], record["bbox_h"]),
                padding_ratio=args.padding_ratio,
                target_size=(224, 224),
            )

        preview_name = f'{record["image_id"]:05d}.jpg'
        build_preview(annotated, normalized, preview_dir / preview_name)
        prefill = prefill_state.get(int(record["image_id"]), {})

        csv_row = {
            "image_id": record["image_id"],
            "relative_image_path": record["relative_image_path"],
            "species_id": record["species_id"],
            "species_name": record["species_name"],
            "split": record["split"],
            "bbox_x": record["bbox_x"],
            "bbox_y": record["bbox_y"],
            "bbox_w": record["bbox_w"],
            "bbox_h": record["bbox_h"],
            "crop_left": crop_box[0],
            "crop_top": crop_box[1],
            "crop_right": crop_box[2],
            "crop_bottom": crop_box[3],
            "preview_file": preview_name,
            "review_index": review_index,
            "perching_like": record["perching_like"],
            "upright_water_like": record["upright_water_like"],
            "tree_clinging_like": record["tree_clinging_like"],
            "one_eye_visible": record["one_eye_visible"],
            "one_wing_visible": record["one_wing_visible"],
            "legs_visible": record["legs_visible"],
            "bbox_horizontal": record["bbox_horizontal"],
            "assignment_score": record["assignment_score"],
            "keep": prefill.get("keep", ""),
            "notes": prefill.get("notes", ""),
        }
        csv_rows.append(csv_row)
        gallery_cards.append(csv_row)

    csv_fields = [
        "image_id",
        "relative_image_path",
        "species_id",
        "species_name",
        "split",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "preview_file",
        "review_index",
        "perching_like",
        "upright_water_like",
        "tree_clinging_like",
        "one_eye_visible",
        "one_wing_visible",
        "legs_visible",
        "bbox_horizontal",
        "assignment_score",
        "keep",
        "notes",
    ]
    write_csv(output_dir / "candidates.csv", csv_rows, csv_fields)
    render_gallery(gallery_cards, output_dir / "gallery.html")

    print(f"Review CSV:  {output_dir / 'candidates.csv'}")
    print(f"Review HTML: {output_dir / 'gallery.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
