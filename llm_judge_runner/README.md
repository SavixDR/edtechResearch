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

- `run_experiments.py` – live collection entry point.
- `pilot_cost_estimator.py` – token counting and cache-aware cost projection with no model generation.
- `prepare_cases_from_legacy_csv.py` – converts the CSV structure used by the older script into baseline T1 cases.
- `src/providers/` – provider-specific API adapters.
- `config/models.json` – your seven judge models and provider settings.
- `config/pricing.json` – editable USD rates and token-count proxy safety multipliers.
- `config/settings/settings_pilot_t0.json` – isolated T0 pilot settings.
- `config/prompt_variants.json` – canonical prompt plus any T2 prompt-order variants.
- `config/assignments.json` – assignment → KC codebook file mapping.
- `data/experiment_cases.jsonl` – canonical experiment input format.
- `outputs/dryruns/<run_id>/` – configuration/prompt validation runs with no generation calls.
- `outputs/live_runs/<run_id>/` – API-enabled result collections and incurred-cost reports.
- `outputs/pilots/<run_id>/` – token-counting and projected-cost pilots with no generation calls.

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

The per-conversation user template is `assets/prompts/judge_v3_user.txt`. It contains `{{OUTPUT_SCHEMA}}` and
`{{DYNAMIC_CASE_INPUT}}`. Python inserts the exact configured schema into the stable prefix, then appends the case metadata
and conversation as the dynamic suffix. The same loaded schema object is used for local validation. Keeping the long,
repeated instructions before the changing conversation also gives provider prefix caching the best chance to apply.

Optional assignment material can be configured per assignment using either inline text or a file:

```json
{
	"codebook_file": "../data/KC/f24_a3/f24_a3_corrected_kc_model.json",
	"assignment_reference_file": "assets/references/f24_a3.txt",
	"human_validated_few_shot_examples_file": "assets/examples/f24_a3.jsonl"
}
```

Omit those optional keys to render empty reference/example sections.

## T0 token and cost pilot (no generation)

Put the real baseline cases in `data/t0_base_cases.jsonl`, using `"test_id": "T0"`. Configure each assignment's
codebook and optional human-validated examples in `config/assignments.json`. No supplementary assignment reference file is
needed for T0; simply omit `assignment_reference` and `assignment_reference_file`.

Before calculating USD amounts, enter the selected model's prices in `config/pricing.json`. Rates are per one million
tokens:

```json
"gpt_5_6_sol": {
  "input_per_million": 0.0,
  "cached_input_per_million": 0.0,
  "cache_write_per_million": null,
  "output_per_million": 0.0,
  "token_counter": {
    "reference_model": "gpt-5.6-sol",
    "conservative_multiplier": 1.0
  }
}
```

Use the actual provider prices and record their date in `prices_as_of`. A missing price remains `null`: token totals are
still reported, but the pilot does not invent a cost. For non-OpenAI targets, the default multiplier is 1.4 because the
OpenAI count is only a proxy; edit it if your own comparison supports a different margin.

Offline-only analysis performs no API calls:

```bash
python pilot_cost_estimator.py --model gpt_5_6_sol --mode offline-estimate
```

Provider-count analysis calls OpenAI's input-token count endpoint but never a generation endpoint:

```bash
python pilot_cost_estimator.py --model gpt_5_6_sol --mode provider-count
```

Run both (the default) to retain the offline comparison alongside the provider count:

```bash
python pilot_cost_estimator.py --model gpt_5_6_sol --mode both
```

The pilot is intentionally one model per command. It refuses to start unless
`config/settings/settings_pilot_t0.json` contains `"generation_enabled": false`, and its report records
`"generation_api_calls": 0`. It keeps the fixed 8,000-token live response cap and reports the corresponding worst-case
output cost. The cap does not cause the model to emit 8,000 tokens; it is an upper limit. Actual output cost should be
re-estimated from a later small pilot's observed output lengths.

Each pilot directory contains:

- `token_counts.jsonl` and `token_counts.csv` – full, stable-prefix, and dynamic token estimates per case.
- `pilot_summary.json` – no-cache upper bound and cache-hit scenarios in USD.
- `cache_scenarios.csv` – 0%, 50%, 90%, and 100% assumed stable-prefix hit scenarios.
- `pilot_manifest.json` – settings, pricing, and schema hashes.
- `cases_snapshot.jsonl` and `prompt_snapshots/` – exact inputs used for the estimate.

The cache figures are projections, not observed cache performance. Token counting cannot establish that a provider will
accept, retain, or hit a cache. Stable prompt prefixes are counted separately from dynamic case text, the first request in
each assignment/prompt group is treated as populating the cache, and subsequent stable-prefix tokens are divided between
cached and uncached rates according to the selected scenario. Provider-specific storage charges are not included.

### T2 prompt/order variants

If you create a shuffled Task/KC/mapping-order version of the V3 prompt, save it as another text file and add it to
`config/prompt_variants.json`, for example:

```json
{
	"canonical": { "file": "assets/prompts/judge_v3_canonical.txt" },
	"t2_kc_order_001": { "file": "assets/prompts/judge_v3_t2_kc_order_001.txt" }
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
	"chunk_id": "conv_0042::chunk_001",
	"chunk_index": 1,
	"chunk_count": 3,
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
		{ "turn_index": "1.student", "speaker": "student", "text": "..." },
		{ "turn_index": "1.assistant", "speaker": "assistant", "text": "..." }
	]
}
```

The key field for later pairing is `base_conversation_id`. All altered versions of the same source conversation retain
that value.

### Pre-chunked conversations

Long conversations should be split deterministically before collection, with one JSONL case per chunk. Every chunk keeps
the same `conversation_id` and `base_conversation_id`, while `chunk_id` and `case_id` must be unique for that chunk:

```text
conversation_id: chat_123
├── case_id: f24_a2::chat_123::chunk_001::T0::base
│   ├── chunk_id: chat_123::chunk_001
│   ├── chunk_index: 1
│   └── chunk_count: 2
└── case_id: f24_a2::chat_123::chunk_002::T0::base
    ├── chunk_id: chat_123::chunk_002
    ├── chunk_index: 2
    └── chunk_count: 2
```

`chunk_index` and `chunk_count` are optional, but they must be supplied together, must be positive integers, and the index
cannot exceed the count. For an unchunked conversation, either omit them or use `1` for both. Older inputs without a
`chunk_id` remain supported: the loader uses `conversation_id` as the default.

The runner sends and stores each chunk independently. Resume tracking therefore operates per unique `case_id`. Results
remain linked through `conversation_id`, but the collection package does not merge chunk annotations; conversation-level
aggregation should be performed later using an explicitly defined analysis rule.

Prepared JSONL cases can be chunked reproducibly at complete student-assistant exchange boundaries:

```bash
python prepare_chunked_conversations.py \
  data/balanced_conversations_24.jsonl \
  data/balanced_conversations_24_chunked.jsonl \
  --max-characters 12000
```

The utility never splits or overlaps an exchange. It preserves singleton identifiers, gives split chunks unique IDs,
adds one-based `chunk_index`/`chunk_count` fields, records the policy in `transformation_metadata.chunking`, validates
exact turn reconstruction, and writes the output atomically. Existing output files require `--overwrite`.

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

### Interactive setup

Run the collector without execution arguments to open the guided questionnaire:

```bash
python run_experiments.py
```

The questionnaire first lists JSON files under `config/settings/` and asks which settings file to use. It accepts the
displayed number, a relative path, an absolute path, or blank for `config/settings/settings.json`. The selected file must
exist and load as a valid runner configuration before setup continues. It then asks for the execution mode, model keys,
test IDs, optional exact case IDs, repeat count, and run ID. Dry run is the default. A live run, including a resumed live
run, displays its planned request count and output directory and proceeds only when `LIVE` is entered exactly. Resume
additionally asks whether previously recorded API errors should be retried.

Supplying only `--settings path/to/settings.json` opens the remaining questionnaire using that configuration without
asking for the settings file again.

If any execution argument such as `--models`, `--tests`, `--dry-run`, `--run-id`, or `--resume` is supplied, the command
remains non-interactive for scripts and automated workflows.

### Direct dry-run command

```bash
python run_experiments.py --dry-run
```

A dry run resolves cases, prompt variants, codebooks, hashes, model configuration, and writes a run manifest. It does
not call a model. Dry-run artifacts are written under `outputs/dryruns/`, separately from API-enabled runs.

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
pricing, repeats, or output settings in one run directory. API-enabled runs are resumed from `outputs/live_runs/`;
dry runs are resolved under `outputs/dryruns/` when `--dry-run` is supplied.

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

Every run produces:

- `run_manifest.json` – frozen experiment/model/prompt/codebook provenance.
- `cases_snapshot.jsonl` – exact selected cases.
- `prompt_snapshots/` – exact rendered prompts sent during that run.

An API-enabled run under `outputs/live_runs/<run_id>/` also produces:

- `results.jsonl` – full structured research records.
- `results_flat.csv` – evaluation-friendly flat table.
- `errors.jsonl` – only API/parse/schema failures.
- `costs.jsonl` – normalized token categories and calculated cost for each collected request record.
- `cost_summary.json` – run total plus model, test, and response-status breakdowns.
- `cost_summary.csv` – compact overall and per-model summary rows.

Cost reports are refreshed after every completed request, so completed costs remain available after interruption. They
use provider-reported token usage and the rates frozen from `config/pricing.json` into the run manifest. If usage or a
required rate is unavailable, the report marks the cost incomplete instead of treating the missing amount as zero.
Provider invoices can still differ because of taxes, rounding, special pricing tiers, or failed requests for which no
usage was returned.

The checked-in historical directories `run_schema_prompt_validation`, `run_schema_prompt_validation_v2`, and
`run_resume_validation` under `outputs/dryruns/` are no-generation validation artifacts. The first captured the initial
schema/prompt integration, `v2` captured the later split system/user snapshot layout, and the resume validation captured
the resume-signature behavior. Their manually chosen run IDs are descriptive labels, not special runner modes.

A single result record contains:

```text
experiment
  test_id / variant_id / variant_type / baseline flag / transformation metadata
case
  case_id / base_conversation_id / conversation_id / chunk_id / chunk position / assignment_id / case hash
judge
  provider / exact configured model ID / adapter / reasoning effort / repeat
input_provenance
  prompt hashes / conversation hash / codebook hash / schema hash
response
  status / raw text / parsed JSON / parse method / validation errors
  provider response id / usage / finish reason / latency / attempts / API error
cost
  normalized uncached/cached/cache-write/output tokens / frozen rates
  component costs / known cost / complete total / completeness warnings
```

This is intentionally richer than a simple `model -> labels` file so you can later evaluate bias without losing the
experimental provenance needed to defend the methodology.

## Important experimental choice: prompt-only JSON by default

The collector does **not** turn on provider-specific JSON-schema enforcement by default. This is deliberate. Your T7/T8
suite includes prompt-injection and output-format attacks. If one provider were protected by a strict API-level schema
while another were not, that could confound the robustness comparison.

Instead, every model receives your frozen prompt's JSON requirement, the raw response is always preserved, and local
parsing is controlled by `parse_mode`:

- `strict` – accept only a top-level JSON object with no Markdown fences or surrounding text. This measures exact
  output-format compliance.
- `tolerant` – first try direct JSON, then accept a JSON object in a Markdown code fence or surrounded by prose. The
  repair method is recorded, so recovered output can be analysed separately from direct JSON.

Parsing is separate from schema/context validation. A tolerant response can still receive `schema_error` when required
fields, enum values, task IDs, KC IDs, ordering, or links are invalid. In both parse and schema failures, the original
`raw_text` is retained; tolerant parsing additionally retains the recovered object in `parsed_json`.

The recorded parse methods are:

- `direct_json`
- `json_code_fence` (only with tolerant mode)
- `extracted_json_object` (only with tolerant mode)
- parse error

`config/settings/settings.json` and the token-counting pilot default to `"parse_mode": "strict"`. The live T0 settings
use `"parse_mode": "tolerant"` so presentation wrappers do not discard usable annotations. For a new run, choose the
mode deliberately and keep the resulting `run_manifest.json` with the data.

## V3 output-schema validation

The schema used by both the V3 prompt and validator is configured as:

```json
"output_schema_file": "assets/judge_v3_output_schema.json"
```

It requires only `tasks`, `labels`, and `flags`; constrains confidence to 0-1; limits evidence to one or two non-empty
strings; and enforces the frozen status and flag values. `conversation_id`, `chunk_id`, and `assignment_id` are already
known to Python and are attached to the stored result record instead of being regenerated by the model. The collector also
checks that every returned task/KC ID exists in the selected assignment's frozen codebook. Invalid responses receive
`schema_error` status, while their raw text is still preserved.

## Model IDs and endpoints

`config/models.json` deliberately keeps API model IDs editable. Model names shown in papers/UI and actual API IDs can differ,
and provider aliases can move. Freeze the exact IDs available in your own provider account before the final experiment and
keep the resulting `run_manifest.json` with your research data.

### Anthropic prompt caching

The Anthropic adapter supports an explicit cache breakpoint on the rendered system prompt through the model's `extra`
configuration:

```json
"extra": {
  "cache_control": {
    "type": "ephemeral",
    "ttl": "5m"
  }
}
```

This caches the stable assignment instructions, task definitions, KC codebook, mapping, and few-shot examples while leaving
the conversation-specific user message uncached. The explicit system breakpoint is intentional: the final user block changes
between cases, so placing an automatic breakpoint at the end of the request would not reliably write the reusable prefix.
Anthropic reports cache writes and reads as `cache_creation_input_tokens` and `cache_read_input_tokens`; the live cost report
records them separately. The five-minute lifetime starts when a cache entry is written or read, and a cache hit refreshes it.
The adapter consumes the Messages API as a stream and reconstructs the final message before parsing, which keeps long-lived
connections active without changing the stored output format. Anthropic SDK retries are disabled so that the runner's
`max_api_attempts` setting is the only retry controller and each recorded attempt maps to one provider request.

### DeepSeek streaming and automatic caching

DeepSeek uses the OpenAI-compatible Chat Completions adapter. Set `"stream": true` on its entry in `config/models.json`
to consume the answer as SSE events; the adapter reconstructs only `delta.content` as the judge output and captures usage
from the final stream chunk. Thinking-mode `reasoning_content` is consumed but is not mixed into the JSON answer. Set
`"sdk_max_retries": 0` so that the runner's `max_api_attempts` remains the only retry controller.

DeepSeek context caching is automatic and requires no request parameter. The cost normalizer reads the provider's
`prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` fields, so cached and uncached input costs remain separate in
the live cost reports.

### Qwen streaming and automatic caching

Qwen3.8-Max uses the international DashScope OpenAI-compatible endpoint. Its model entry enables SSE streaming and
disables SDK retries, leaving `max_api_attempts` as the runner's only retry controller. The adapter reconstructs only
answer content; thinking content is not mixed into the JSON output. It requests final-stream usage so Qwen's automatic
implicit-cache hits can be read from `prompt_tokens_details.cached_tokens` and priced separately.

`reasoning_effort` is intentionally unset. This preserves Qwen3.8-Max's provider-default thinking configuration rather
than translating an explicitly supplied `xhigh` value into a larger reasoning budget. The combined reasoning-and-answer
cap is sent using `max_completion_tokens`.
