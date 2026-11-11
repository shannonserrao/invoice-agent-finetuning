# Reviewer guide

You do not need to run any code. You will receive `review_queue.csv`.

## What to do
1. Open the file in Excel or Google Sheets.
2. For each row, look at `input_ref` (the document image path; open it beside the predicted JSON) and at `prediction`.
3. Fill in:
   - `verdict`: `correct`, `incorrect` or `unsure`.
   - `corrected`: only when the verdict is `incorrect`. Enter the full corrected JSON for the document (must validate against `schema/document.schema.json`).
   - `reviewer`: your name or initials.
   - `note`: optional, anything that would help the next reviewer.
4. Do not edit the other columns. Leave a row blank if you did not get to it.
5. Save as CSV named `reviewed.csv` and send it back.

## Guidance
- Use `unsure` when the document itself is ambiguous. Do not guess.
- The `reason` column tells you why the row was chosen. Treat both kinds the same way.
- If you see the same mistake several times, say so in `note`; it probably needs a guideline change.

## For the engineer
```bash
make feedback-export          # writes feedback/review_queue.csv
make feedback-ingest          # reads feedback/reviewed.csv
git add feedback/corrections.jsonl && git commit -m "data: reviewed batch <date>"
```
The rules are in `specs/05-human-feedback.md`.
