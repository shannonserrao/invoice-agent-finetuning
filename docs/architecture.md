# Architecture — Invoice and Receipt Extraction

The diagrams are Mermaid and render on GitHub and GitLab.

## 1. Training and evaluation pipeline
What `make data train predict eval` does, and what each step writes.

```mermaid
flowchart LR
    subgraph Sources
        CORD[(CORD receipts<br/>pinned revision)]
        SROIE[(SROIE receipts<br/>second test set)]
        COR[(feedback/<br/>corrections.jsonl)]
    end
    subgraph Data["make data"]
        M1[Dataset mappers] --> M2[Canonical JSON<br/>schema/document.schema.json]
        M2 --> M3[Checks: schema, arithmetic,<br/>split leakage]
        M3 --> MAN[data/manifest.json<br/>data-vN + SHA-256]
    end
    subgraph Train["make train"]
        T1["Image + prompt<br/>to target JSON"] --> T2[LoRA fine-tune<br/>small vision-language model]
        T2 --> T3[Keep best adapter by<br/>validation field F1]
    end
    subgraph Eval["make predict / eval"]
        P1[Generate JSON on test] --> P2[Parse, validate,<br/>confidence, route]
        P2 --> E1[Field F1, key fields,<br/>routing quality, cost]
        BL[Zero-shot baseline<br/>expensive model, cached] --> E1
    end
    CORD --> M1
    SROIE --> M1
    COR --> M2
    MAN --> T1
    T3 --> P1
    E1 --> REP[artifacts/report.json]
    T2 -. params, loss .-> ML[(MLflow)]
    E1 -. metrics, report .-> ML
    T3 -. adapter .-> REG[(Model registry)]
```

## 2. Inference path: extract, check, route
How one document is handled at run time.

```mermaid
flowchart TD
    IMG[Document image] --> VLM[Small vision-language model<br/>+ LoRA adapter, in-house]
    VLM --> JSON[Constrained JSON output<br/>+ token log-probabilities]
    JSON --> CHK{Schema valid?<br/>Line items sum to subtotal?<br/>Subtotal + tax = total?}
    CHK -- any fail --> REV
    CHK -- all pass --> CONF{"confidence at or above<br/>threshold?"}
    CONF -- yes --> AUTO[route = auto<br/>straight-through]
    CONF -- no --> REV[route = review]
    REV --> BIG[Optional: expensive model retry]
    BIG --> CHK2{Checks pass?}
    CHK2 -- yes --> AUTO2[Accept escalated result]
    CHK2 -- no --> HUM[Human review queue]
    AUTO --> LOG[(predictions_log.jsonl)]
    AUTO2 --> LOG
    HUM --> LOG
```

## 3. Human feedback loop

```mermaid
flowchart LR
    LOG[(predictions_log.jsonl)] --> SEL[Select: all review-routed<br/>+ random audit of auto]
    SEL --> CSV[review_queue.csv]
    CSV --> HUM[Reviewer compares image<br/>with JSON, corrects]
    HUM --> ING[make feedback-ingest<br/>validates rows]
    ING --> COR[(corrections.jsonl<br/>committed to git)]
    COR --> DATA[make data<br/>new data version]
    DATA --> TRAIN[Retrain adapter: challenger]
    ING --> AUD[Audit agreement on auto]
    AUD --> GATE{Promotion gate met?}
    TRAIN --> GATE
    GATE -- yes --> CH[Promote to champion]
    GATE -- no --> KEEP[Keep current champion]
```

## 4. Model composition
One frozen base model, small trainable adapters. The stretch task adds a second adapter and a router.

```mermaid
flowchart LR
    BASE[Frozen base model<br/>2–4B parameters] --> A1[LoRA adapter: receipts]
    BASE --> A2[LoRA adapter: invoices<br/>stretch T-17]
    R[Adapter router<br/>document type] --> A1
    R --> A2
    A1 --> OUT[Canonical JSON]
    A2 --> OUT
```

## 5. Versioning and release flow

```mermaid
flowchart LR
    DEV[Feature branch] --> MR[Merge request]
    MR --> CI[CI on GitHub / GitLab:<br/>lint + unit tests]
    CI --> MAIN[main]
    MAIN --> RUN[Tracked release run<br/>git SHA + data-vN + config]
    RUN --> GATES{Release gates<br/>SPEC section 5}
    GATES -- pass --> TAG[git tag vX.Y.Z]
    TAG --> REG[Register adapter model-X.Y.Z<br/>alias: challenger]
    REG --> REV[Human audit] --> CHAMP[alias: champion]
    GATES -- fail --> DEV
```

## 6. Where it runs

```mermaid
flowchart TB
    REPO[(Git repo:<br/>code, configs, specs, manifest)]
    REPO --> L[Local<br/>configs/base.yaml]
    REPO --> C[Colab<br/>platform/colab.yaml<br/>artifacts on Drive]
    REPO --> K[Kaggle<br/>platform/kaggle.yaml<br/>artifacts in kernel output]
    REPO --> B[Databricks<br/>platform/databricks.yaml<br/>managed MLflow]
    L & C & K & B --> SAME[Same CLI:<br/>data, train, predict, evaluate]
```
