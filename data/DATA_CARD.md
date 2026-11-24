# SafeShare Data Card — Prepared release v1

**Status:** Updated from the executed Notebook 1 `audit.json` and `manifest.json` supplied with this review. This is a documented synthetic-data audit, **not** a manual ground-truth quality certification or a real-world representativeness study.

## Purpose and provenance

SafeShare researches detection of seven sensitive-information categories in text and OCR-derived screenshot text to support **human-reviewed** redaction. Upstream dataset: [`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII), revision `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed`, reported license **CC-BY-4.0**. Confirm attribution and redistribution obligations against the upstream license before publishing any derived data. The tokenizer is `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Data preparation seed: `42`; token limit: 1024.

The source data are synthetic; they do not validate performance on genuine screenshots, live credentials, handwriting, other languages, or arbitrary user interfaces. **Do not commit raw records or test text to the public repository.**

## Active split allocation

The older 10,000/1,000/1,000 local development run is superseded. Current configuration: `all_eligible_grouped`, with an approximately 80/20 development train/validation split and a separate 10,000-document official-source test subset.

| Split | Documents | Related groups | SFT windows | Negative documents |
|---|---:|---:|---:|---:|
| Train | 79,598 | 39,927 | 131,610 | 22,754 (28.6%) |
| Validation | 19,899 | 9,981 | 33,038 | 5,649 (28.4%) |
| Test | 10,000 | 5,012 | 16,188 | 1,934 (19.3%) |

| Entity | Train | Validation | Test |
|---|---:|---:|---:|
| PERSON | 52,893 | 13,235 | 9,717 |
| EMAIL | 39,571 | 9,925 | 5,435 |
| PHONE | 19,873 | 5,060 | 2,364 |
| ACCOUNT_NUMBER | 7,916 | 2,069 | 1,669 |
| PAYMENT_CARD | 7,117 | 1,893 | 1,291 |
| SSN | 3,985 | 932 | 611 |
| PASSWORD | 6,570 | 1,678 | 713 |

Counts above are from the provided executed audit, not inferred from the notebook defaults. The upstream source has 100,000 records each in the original train and test partitions. Rejections for `in_scope_surface_mismatch`: **494 train**, **265 test**. One exact duplicate was removed from the original train partition; **8** train rows were excluded for overlap with official-test groups. The audit records **0** duplicate-conflict removals and **0** cross-split overlaps for ID, group ID, source UID, exact text hash, normalized text hash, and template hash. This does not rule out semantic near-duplicates or hidden shared personas.

## Labels and annotation policy

| Source annotation | SafeShare label |
|---|---|
| `account_number` | `ACCOUNT_NUMBER` |
| `credit_debit_card` | `PAYMENT_CARD` |
| `email` | `EMAIL` |
| `first_name` | `PERSON` |
| `last_name` | `PERSON` |
| `password` | `PASSWORD` |
| `phone_number` | `PHONE` |
| `ssn` | `SSN` |

Other upstream annotations are **out of scope**, not necessarily nonsensitive. Original annotations are character-offset `[start, end)` spans; model targets are zero-based **inclusive** word-ID ranges. The word mapping conservatively expands partially overlapping character spans to whole non-whitespace words, potentially increasing mask size. Neighboring name spans may be consolidated. Supervised SFT windows are not valid evaluation windows because they can depend on gold target length.

## Reproducibility and integrity

- Prepared release receipt/run: `20261109T224019Z-fe313d60` (from uploaded test filename and existing run context); manifest `release_id`: `safeshare-dd983807cd5d6d89`; schema `safeshare-data-v1`.
- GCS prepared-data prefix: `gs://ai-caps-proj26-bucket/safeshare/data/prepared/20261109T224019Z-fe313d60` (confirm against the Notebook 1 receipt before using it as an authoritative location).
- Frozen **test document file**: `docs/test.jsonl`, 10,000 rows, 81,226,057 bytes.
- SHA-256 of `docs/test.jsonl`: `51fbb4843e6b5f568b9d45cfea2af77cdba491d76109f7ed5ecb06b8f611bd3c` — **independently recomputed on the supplied uploaded file; matches manifest**.
- SFT test hash: `c300031878cfc342a3041bf23c0399a5115311473666d3b33f85c10fa256a4f5` (manifest-reported, not independently recomputed).
- Tokenized test hash: `5275dc591c393a63472b32f905cfb3779ca923888d2ae5229eed41c352c64cbb` (manifest-reported, not independently recomputed).
- `audit.json` manifest hash: `6d804e5c5f5ab10e7a885262c49cf854a3247972ebdeba917050a20b6968d615` (manifest-reported; independent audit-file hash verification is a follow-up).
- For all other artifact hashes, see the original `manifest.json`; do not treat metadata-only hash values as independent verification.

**Test freeze:** preserve this exact test file, manifest, and hashes. The test set is for one final locked evaluation only. Earlier model smoke tests used validation subsets. The uploaded test file was read only to compute its hash and size during this documentation update; no test model predictions were generated.

## Screenshot/OCR scope and limitations

Notebook 1 also prepares synthetic screenshot images derived from source documents; images inherit their source split and group. Image counts and OCR accuracy must be sourced from the **image-specific** manifest/audit and receipt, which were not included among the two JSON files supplied for this documentation update. Do not extrapolate text split counts to image counts. OCR transcription and box overlap do not measure sensitive-region redaction quality. A separate manually labeled real-world screenshot challenge set remains outstanding.

## Remaining audit work / limitations

1. Manual sample-based label-quality inspection, including word-boundary expansion and false negatives for excluded labels.
2. Verify original source licensing, attribution, and synthetic-data generation limitations against the upstream dataset card.
3. Independently verify the original audit file hash and all needed image release hashes against their own manifests.
4. Measure real-screenshot and multilingual performance before claiming generalizability.
5. Maintain restricted GCS access and avoid recording unredacted inputs, passwords, or full credentials in application logs.

**Intended use:** research, evaluation, and human-reviewed sensitive-data masking. **Not intended:** guaranteed anonymization, automated compliance certification, or sharing without user review.
