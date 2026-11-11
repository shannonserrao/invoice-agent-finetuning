# Data card — Invoice and Receipt Extraction

Fill in the blanks during T-02 to T-05. Do not publish results before the licence rows are complete.

| Field | CORD | SROIE |
|---|---|---|
| Content | Receipt images with structured labels | Receipt images with four header fields |
| Hub ID or source, revision | `naver-clova-ix/cord-v2` @ `7f0115a4b758a71d6473b8d085751692da2fef98` | `jsdnrs/ICDAR2019-SROIE` @ `bffe40c26759f3376ec2b3ae9031dbba54cd587c` (community re-release with 14 extra test annotations) |
| Licence | CC-BY-4.0 (Hub metadata) | CC-BY-4.0 (Hub metadata and dataset card) |
| Published splits | 800 / 100 / 100 (confirmed, T-02) | 626 train / 361 test, no validation. Only the published test split is used, as `test_sroie.jsonl` (evaluation only, never training). |
| Languages and currency | Indonesian receipts, amounts like `1,346,000` (currency not labelled) | English, Malaysian receipts, amounts in RM (currency not labelled, so null) |
| Fields mapped to canonical schema | line_items (from `menu`: nm, cnt, unitprice, price), subtotal (`subtotal_price`), tax (`tax_price`), total (`total_price`) | vendor_name (company), date, total. Address is not in the schema. |
| Fields not labelled (set to null, excluded from metrics) | vendor_name, date, currency (CORD has none of these; always null) | line_items, subtotal, tax, currency |
| Gold arithmetic-consistency rate | data-v1, tolerance 0.01: `total_ok` train 0.676 / val 0.703 / test 0.683; `items_sum_ok` train 0.851 / val 0.815 / test 0.871. Total fails often because `service_price`, `discount_price` and `etc` have no canonical field and are not folded into tax. | n/a (no line items or subtotal) |
| Personal data | Receipts may show shop addresses and staff names; check before sharing samples | same |

## Known limits
- Receipts, not full invoices. Invoices have longer line-item tables, payment terms and tax identifiers.
- Limited geography and store types; layouts in your documents will differ.
- Small test sets: about 100 documents means wide confidence intervals.
- Label conventions differ between datasets; the mapper documents each choice.

## Date handling (SROIE)
Dates are parsed to ISO 8601 by `data.parse_date`. All-numeric dates are read as day/month or month/day only when one part exceeds 12; otherwise they are ambiguous and set to null, as spec 01 §3 requires. In data-v1, 125 of 361 SROIE test dates (35%) are null for this reason. Date accuracy on SROIE is therefore measured on the 236 documents with an unambiguous gold date.

## Changes
| Data version | Date | Change |
|---|---|---|
| data-v1 | 2026-11-11 | Initial mapping and split |
| data-v1-fb<fingerprint> | | Created by `make data` whenever `feedback/corrections.jsonl` contributes training rows (T-13). See `data/manifest.json` for the applied and excluded ids. |
