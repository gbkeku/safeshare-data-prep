# SafeShare responsible-AI review: data preparation (Notebook 1)

Scope: the data used to train and evaluate SafeShare, as built and documented in Notebook 1 (data run `20261111T013302Z-e38b2785`). Model behaviour, the app and the live service are covered in [RESPONSIBLE_AI.md](RESPONSIBLE_AI.md). This review records what the data can and cannot support, so later claims about the model rest on known ground.

## 1. Source, licence and attribution

- **Dataset:** [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII), loaded at a pinned revision (`b70ffaf5…`) so every run uses identical data.
- **Licence:** CC BY 4.0, which allows reuse and modification with attribution. Notebook 1 saves an `ATTRIBUTION.txt` with the raw copy, and the README credits the dataset and its licence.
- **Provenance:** an untouched copy of the source is kept with SHA-256 hashes, and every later file is tied to it by a manifest. Any figure in the project can be traced back to the exact source records.

## 2. Privacy

- **The records are synthetic.** No real person's data was collected, so no consent or data-subject rights apply to the training data. The names, emails and numbers are realistic by design, and some may by chance match real people or accounts, so the data is still handled as if it were sensitive.
- **Storage:** the raw and prepared data are kept in a private Google Cloud Storage bucket, never in the public repository. The repository holds only metadata: the data card, frozen hashes and the receipt.
- **Screenshots** are rendered from the synthetic text. No real screenshots of real people's information were used anywhere in data preparation.
- **Minimisation:** only the seven categories SafeShare detects are labeled for training. Other personal information in the records is left as ordinary text, not extracted or collected separately.

## 3. Label quality

- **Validation before use:** a record is removed whole if any annotation is out of bounds, empty, or (for the seven categories) does not match the text it points to. Records are never repaired by guessing, so an uncertain label cannot enter training. Notebook 1, section 10 shows how many records were removed and why.
- **Known noise that passes these checks:** the Week 4 error review found labels that point at the right text but are wrong in meaning. Field names were labeled as passwords ("Biometric Identifier for Access to Production Files"), a budget figure was labeled as a card number, and real-looking card numbers, CVVs, emails and a person's name were left unlabeled. These come from the dataset itself. Their effect is that measured leak and over-mask rates are approximate in both directions. They were left unchanged for this release and are reported, not hidden; measuring how common they are is the first planned improvement.
- **Scope decisions are documented:** the label-mapping report (section 9) lists every source label, which of the eleven are mapped to the seven categories, and how often each unmapped label occurs.

## 4. Representativeness and bias

| Risk | What the data shows | Consequence |
|---|---|---|
| **Synthetic, not real** | All text is generated, and the screenshots are rendered from it, with cleaner layouts than real apps | Every measured rate is likely better than real-world performance. A small hand-made set of real-app screenshots with fake data is planned as a reality check |
| **Locale** | US and international versions of each record, in nearly equal numbers | A locale gap can be measured (Week 3 and 4 found none clearly). Finer groups cannot |
| **Names from different backgrounds** | The dataset does not record the cultural origin of names | Whether PERSON detection is worse for some kinds of names **cannot be measured with this data**. This is a known gap, not evidence of fairness |
| **Domain** | Many subject areas, unevenly represented | Some domains are thin. Cybersecurity was later found to leak clearly more than average |
| **Language** | English only | Non-English text is out of scope, and the app's limits notice says so |
| **Out-of-scope personal data** | Street addresses, dates of birth and many other types are unmapped | The trained model does not detect them. Users are told in the app, so they do not assume a screenshot is fully clean |

## 5. OCR as part of the data

Tesseract turns screenshots into the text the model reads, so OCR errors become data errors.

- **Uneven by category.** On validation screenshots Tesseract reads passwords, PINs and API keys exactly only about 60% of the time, against 89% to 99% for other categories, because they mix symbols and look-alike characters. The model is therefore least reliable on some of the most damaging secrets, before any modelling choice is made.
- **A word OCR misses cannot be redacted.** Tesseract finds about 97% of words and locates 99.96% of sensitive spans on these rendered screenshots; real, blurred or small screenshots will do worse.
- **Variety by design.** Rendering varies font, size, theme, layout, scale and blur, and the test split holds out one font and one layout never seen in training.

## 6. Evaluation integrity

- **No leakage:** records are grouped by source, normalized text and masked template, whole groups go to one split, and training records sharing a group with any test record are removed. Section 10 shows zero overlap between splits.
- **Frozen test split:** the test documents and screenshot records were hashed and frozen in EVAL.md before any model was tuned. Every choice since (thresholds, model selection, the review band) was made on validation. The test split has not been scored.
- **Reproducibility:** Notebook 1 rebuilds the data from the pinned source and compares the rebuilt test split with the frozen hashes (section 10).

## 7. Misuse of the data and outputs

The prepared data is a large, labeled set of realistic personal-information patterns. It could help train a tool that **extracts** personal data instead of hiding it. The risk is low because the source dataset is already public, but the prepared data stays in a private bucket and is not redistributed. Only metadata and charts are shared.

## 8. Cost

Data preparation runs on CPU, with no GPU: about 2 hours per run, most of it OCR on about 5,000 screenshots. Outputs are reused across all later notebooks instead of being regenerated.

## 9. Open items

1. Measure the label noise on a random sample of validation errors, and report adjusted leak and over-mask rates.
2. Build the hand-made real-app screenshot set (fake data only) to estimate the synthetic-to-real gap.
3. If a later data release is made, add validity checks for the noise patterns found (field names labeled as secrets, card or email patterns left unlabeled), and review flagged records by hand rather than relabeling them automatically.
