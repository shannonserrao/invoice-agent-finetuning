/* Pure logic for the review page: schema check, edit detection, CSV export.
   Mirrors docextract.schema.schema_errors and docextract.feedback.QUEUE_FIELDS (kept in sync by tests/test_review.py). */
(function (root) {
  "use strict";

  const QUEUE_FIELDS = ["id", "input_ref", "prediction", "confidence", "model_version",
                        "reason", "verdict", "corrected", "reviewer", "note"];
  const VERDICTS = ["correct", "incorrect", "unsure"];
  const HEADER = ["vendor_name", "date", "currency", "subtotal", "tax", "total"];
  const NUMBER_FIELDS = ["subtotal", "tax", "total"];
  const ITEM = ["description", "quantity", "unit_price", "amount"];
  const ITEM_NUMBERS = ["quantity", "unit_price", "amount"];
  const isNumber = (v) => typeof v === "number" && Number.isFinite(v);
  const isPlainObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

  function schemaErrors(doc) {
    if (!isPlainObject(doc)) return ["document is not an object"];
    const errors = [];
    const allowed = HEADER.concat(["line_items"]);
    Object.keys(doc).forEach((k) => { if (!allowed.includes(k)) errors.push("unexpected key '" + k + "'"); });
    allowed.forEach((k) => { if (!(k in doc)) errors.push("missing key '" + k + "'"); });
    HEADER.forEach((k) => {
      const v = doc[k];
      if (v === null || v === undefined) return;
      if (NUMBER_FIELDS.includes(k) && !isNumber(v)) errors.push(k + " must be a number or null");
      if (!NUMBER_FIELDS.includes(k) && typeof v !== "string") errors.push(k + " must be a string or null");
    });
    const items = doc.line_items;
    if ("line_items" in doc && !Array.isArray(items)) errors.push("line_items must be a list");
    (Array.isArray(items) ? items : []).forEach((item, i) => {
      if (!isPlainObject(item)) { errors.push("line_items[" + i + "] is not an object"); return; }
      Object.keys(item).forEach((k) => {
        if (!ITEM.includes(k)) errors.push("line_items[" + i + "] unexpected key '" + k + "'");
      });
      ITEM.forEach((k) => {
        if (!(k in item)) errors.push("line_items[" + i + "] missing key '" + k + "'");
      });
      ITEM_NUMBERS.forEach((k) => {
        const v = item[k];
        if (v !== null && v !== undefined && !isNumber(v)) {
          errors.push("line_items[" + i + "]." + k + " must be a number or null");
        }
      });
      const d = item.description;
      if (d !== null && d !== undefined && typeof d !== "string") {
        errors.push("line_items[" + i + "].description must be a string or null");
      }
    });
    return errors;
  }

  /* Parse editor text. Returns {doc, errors}; doc is null when the text is not JSON. */
  function parseDocument(text) {
    let doc;
    try { doc = JSON.parse(text); } catch (err) { return { doc: null, errors: ["not valid JSON: " + err.message] }; }
    return { doc: doc, errors: schemaErrors(doc) };
  }

  /* Copy with keys in schema order (unknown keys last), so equal documents serialise identically. */
  function ordered(doc) {
    if (!isPlainObject(doc)) return doc;
    const order = ["vendor_name", "date", "currency", "line_items", "subtotal", "tax", "total"];
    const out = {};
    order.forEach((k) => { if (k in doc) out[k] = doc[k]; });
    Object.keys(doc).forEach((k) => { if (!(k in out)) out[k] = doc[k]; });
    if (Array.isArray(out.line_items)) {
      out.line_items = out.line_items.map((it) => {
        if (!isPlainObject(it)) return it;
        const o = {};
        ITEM.forEach((k) => { if (k in it) o[k] = it[k]; });
        Object.keys(it).forEach((k) => { if (!(k in o)) o[k] = it[k]; });
        return o;
      });
    }
    return out;
  }

  /* Stable compact form, so an edit is detected by comparing strings. */
  function canonical(doc) { return JSON.stringify(ordered(doc)); }

  function prettyPrediction(text) {
    try { return JSON.stringify(ordered(JSON.parse(text)), null, 2); } catch (err) { return text; }
  }

  function isEdited(originalText, editorText) {
    const a = parseDocument(originalText), b = parseDocument(editorText);
    if (a.doc === null || b.doc === null) return originalText.trim() !== editorText.trim();
    return canonical(a.doc) !== canonical(b.doc);
  }

  function csvCell(value) {
    const s = value === null || value === undefined ? "" : String(value);
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function toCsv(rows) {
    const lines = [QUEUE_FIELDS.join(",")];
    rows.forEach((r) => lines.push(QUEUE_FIELDS.map((f) => csvCell(r[f])).join(",")));
    return lines.join("\r\n") + "\r\n";
  }

  /* items: queue rows; states: {id: {verdict, text, note}}. Returns {rows, problems}.
     Unreviewed rows (no verdict) are exported blank: ingest skips them. */
  function buildExport(items, states, reviewer) {
    const rows = [], problems = [];
    items.forEach((item) => {
      const s = states[item.id] || {};
      const row = { id: item.id, input_ref: item.input_ref, prediction: item.prediction,
                    confidence: item.confidence, model_version: item.model_version, reason: item.reason,
                    verdict: s.verdict || "", corrected: "", reviewer: s.verdict ? reviewer : "", note: s.note || "" };
      if (s.verdict && !VERDICTS.includes(s.verdict)) problems.push(item.id + ": unknown verdict");
      if (s.verdict && !reviewer.trim()) problems.push(item.id + ": reviewer name is required");
      if (s.verdict === "incorrect") {
        const parsed = parseDocument(s.text === undefined ? "" : s.text);
        if (parsed.errors.length) problems.push(item.id + ": " + parsed.errors.join("; "));
        else row.corrected = canonical(parsed.doc);
      }
      rows.push(row);
    });
    return { rows: rows, problems: problems };
  }

  const api = { QUEUE_FIELDS, VERDICTS, schemaErrors, parseDocument, canonical, prettyPrediction, isEdited,
                csvCell, toCsv, buildExport };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ReviewLogic = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
