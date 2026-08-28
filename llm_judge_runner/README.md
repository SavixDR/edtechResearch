# Modular LLM-as-a-Judge Result Collector

This package replaces the old monolithic `teacher_KC_labels.py` with a modular **collection-only** pipeline.
It does **not** calculate bias, robustness, agreement, flip rate, VSR, or model rankings. It only produces
well-structured, provenance-rich outputs for later evaluation.

## Why the runner is structured this way

Your experiments contain two kinds of controlled variants:

- **Conversation variants** (T3–T8): the prompt remains frozen while the conversation changes.
- **Prompt/order variants** (especially T2): the conversation can remain identical while a different frozen
  prompt variant is selected through `prompt_variant_id`.

Every result therefore stores both the baseline/variant relationship and the exact prompt/codebook hashes.

## Project layout

- `run_experiments.py` – command-line entry point.
- `prepare_cases_from_legacy_csv.py` – converts the CSV structure used by the older script into baseline T1 cases.
- `src/providers/` – provider-specific API adapters.
- `config/models.json` – your seven judge models and provider settings.
- `config/prompt_variants.json` – canonical prompt plus any T2 prompt-order variants.
- `config/assignments.json` – assignment → KC codebook file mapping.
- `data/experiment_cases.jsonl` – canonical experiment input format.
- `outputs/<run_id>/` – immutable result collection for one run.

## 1. Install

The authoritative V3 output contract is `assets/judge_v3_output_schema.json`. The accompanying
`assets/judge_v3_output_example.json` is illustrative only and is not used for validation.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add the API keys/base URLs you actually use.

## 2. Put your exact V3 prompt in the prompt file

Replace the placeholder in:

`assets/prompts/judge_v3_canonical.txt`

The runner intentionally refuses a live run while the placeholder remains. This prevents accidentally collecting
research data with a non-frozen prompt.

The system prompt file may contain:

- `{{ASSIGNMENT_ID}}`
- `{{FROZEN_TASKS}}`
- `{{TASK_KC_MAPPING}}`
- `{{FROZEN_CODEBOOK}}`
- `{{ASSIGNMENT_REFERENCE_MATERIAL}}`
- `{{HUMAN_VALIDATED_FEW_SHOT_EXAMPLES}}`

The per-conversation user template is `assets/prompts/judge_v3_user.txt`. It contains `{{CONVERSATION}}` and
`{{OUTPUT_SCHEMA}}`. The runner inserts the exact configured schema into this user message and uses the same loaded object
for local validation. This prevents the model instructions and Python validator from drifting apart.

Optional assignment material can be configured per assignment using either inline text or a file:

```json
{
  "codebook_file": "../data/KC/f24_a3/f24_a3_corrected_kc_model.json",
  "assignment_reference_file": "assets/references/f24_a3.txt",
  "human_validated_few_shot_examples_file": "assets/examples/f24_a3.jsonl"
}
```

Omit those optional keys to render empty reference/example sections.

### T2 prompt/order variants

If you create a shuffled Task/KC/mapping-order version of the V3 prompt, save it as another text file and add it to
`config/prompt_variants.json`, for example:

```json
{
  "canonical": {"file": "assets/prompts/judge_v3_canonical.txt"},
  "t2_kc_order_001": {"file": "assets/prompts/judge_v3_t2_kc_order_001.txt"}
}
```

Then a T2 case simply uses `"prompt_variant_id": "t2_kc_order_001"`.

## 3. Canonical experiment-case format

Each JSONL row is one exact input condition:

```json
{
  "case_id": "f24_a3::conv_0042::T4::formal_01",
  "base_conversation_id": "conv_0042",
  "conversation_id": "conv_0042",
  "chunk_id": "chunk_001",
  "assignment_id": "f24_a3",
  "student_id": "student_12",
  "test_id": "T4",
  "variant_id": "formal_01",
  "variant_type": "style_bias",
  "is_baseline": false,
  "prompt_variant_id": "canonical",
  "transformation_metadata": {
    "style": "formal",
    "semantic_content_expected_unchanged": true
  },
  "turns": [
    {"turn_index": "1.student", "speaker": "student", "text": "..."},
    {"turn_index": "1.assistant", "speaker": "assistant", "text": "..."}
  ]
}
```

The key field for later pairing is `base_conversation_id`. All altered versions of the same source conversation retain
that value.

Recommended `test_id` values:

- `T1` baseline/base test set
- `T2` position/order
- `T3` length
- `T4` style/sentiment
- `T5` concreteness
- `T6` mapping/familiar-knowledge
- `T7` prompt injection
- `T8` other adversarial attack

The collector does not interpret these labels; it preserves them.

## 4. Convert your old sample CSV to baseline cases

Example:

```bash
python prepare_cases_from_legacy_csv.py path/to/a3_samples.csv data/a3_base_cases.jsonl --assignment-id f24_a3
```

You can then combine those rows with the controlled variant rows you create for T2–T8.

## 5. Validate everything without spending API credits

```bash
python run_experiments.py --dry-run
```

A dry run resolves cases, prompt variants, codebooks, hashes, model configuration, and writes a run manifest. It does
not call a model.

## 6. Select models and tests

All enabled models:

```bash
python run_experiments.py
```

Only GPT + Claude on baseline and prompt injection:

```bash
python run_experiments.py --models gpt_5_6_sol,claude_opus_5 --tests T1,T7
```

One model and exact cases:

```bash
python run_experiments.py --models gemini_3_1_pro --case-ids CASE_001,CASE_002
```

Repeat every condition three times:

```bash
python run_experiments.py --repeats 3
```

Resume a partially completed run:

```bash
python run_experiments.py --run-id run_20260827_main --resume
```

Use the same selection arguments as the original command. For example, if the run started with
`--models gpt_5_6_sol,claude_opus_5 --tests T1,T7 --repeats 3`, include those arguments again when resuming.
The runner verifies a frozen resume signature and refuses to mix changed cases, models, prompts, schemas, codebooks,
repeats, or output settings in one run directory.

You may stop a live run with `Ctrl+C`. Every completed model/case/repeat record has already been appended to
`results.jsonl`. On resume, those execution keys are skipped; an interrupted in-flight request has no completed record
and is run again.

An API failure is recorded after the configured internal retry attempts, and collection continues with later conditions.
By default it is considered a completed research record. To retry conditions whose only recorded status is `api_error`:

```bash
python run_experiments.py --run-id run_20260827_main --resume --retry-api-errors
```

The original API-error record is preserved. The retry is appended with the same `execution_key` and a higher
`collection_attempt`; downstream analysis should select the highest collection attempt for that execution key when a
retry was requested. Parse and schema errors are model-output outcomes and are not retried by this option.

## 7. Output structure

Each run produces:

- `run_manifest.json` – frozen experiment/model/prompt/codebook provenance.
- `cases_snapshot.jsonl` – exact selected cases.
- `prompt_snapshots/` – exact rendered prompts sent during that run.
- `results.jsonl` – full structured research records.
- `results_flat.csv` – evaluation-friendly flat table.
- `errors.jsonl` – only API/parse/schema failures.

A single result record contains:

```text
experiment
  test_id / variant_id / variant_type / baseline flag / transformation metadata
case
  case_id / base_conversation_id / conversation_id / assignment_id / case hash
judge
  provider / exact configured model ID / adapter / reasoning effort / repeat
input_provenance
  prompt hashes / conversation hash / codebook hash / schema hash
response
  status / raw text / parsed JSON / parse method / validation errors
  provider response id / usage / finish reason / latency / attempts / API error
```

This is intentionally richer than a simple `model -> labels` file so you can later evaluate bias without losing the
experimental provenance needed to defend the methodology.

## Important experimental choice: prompt-only JSON by default

The collector does **not** turn on provider-specific JSON-schema enforcement by default. This is deliberate. Your T7/T8
suite includes prompt-injection and output-format attacks. If one provider were protected by a strict API-level schema
while another were not, that could confound the robustness comparison.

Instead, every model receives your frozen prompt's JSON requirement, the raw response is preserved, and local parsing is
recorded as one of:

- `direct_json`
- `json_code_fence` (only with tolerant mode)
- `extracted_json_object` (only with tolerant mode)
- parse error

`config/settings.json` defaults to `"parse_mode": "strict"`.

## V3 output-schema validation

The schema used by both the V3 prompt and validator is configured as:

```json
"output_schema_file": "assets/judge_v3_output_schema.json"
```

It requires `conversation_id`, `chunk_id`, `assignment_id`, `tasks`, `labels`, and `flags`; constrains confidence to 0-1;
limits evidence to one or two non-empty strings; and enforces the frozen status and flag values. The collector additionally
checks that response metadata matches the case and that every returned task/KC ID exists in the selected assignment's
frozen codebook. Invalid responses receive `schema_error` status, while their raw text is still preserved.

## Model IDs and endpoints

`config/models.json` deliberately keeps API model IDs editable. Model names shown in papers/UI and actual API IDs can differ,
and provider aliases can move. Freeze the exact IDs available in your own provider account before the final experiment and
keep the resulting `run_manifest.json` with your research data.
