# schema_gen

Static (no-import) generator: nanoPLM `get-yaml` templates + config dataclasses → canonical
JSON Schema for the docs site (`website/src/generated/`).

## Run

```bash
python -m tools.schema_gen.generate     # writes website/src/generated/*.schema.json + index.json
python -m pytest tests/docs_schema -q   # full test suite incl. the parity gate
```

Dependencies: **stdlib + PyYAML only** — no `torch`, no `nanoplm` import. The configs import
torch, so this generator reads them *statically* (via `ast`/`tokenize`) rather than importing them,
keeping CI light and runnable without the training stack.

## How it works

| module | responsibility |
|--------|----------------|
| `sources.py` | registry of where each surface's docs live (template files / dataclasses), the compat matrix, and the parity allowlist |
| `templates.py` | load a `get-yaml` template (packaged `.yaml` or embedded Python string) and parse its fields + inline comments |
| `dataclass_ast.py` | extract `@dataclass` fields + leading/trailing comments (for the `eval` surface + parity) |
| `types.py` | infer a JSON-Schema type/default from a raw YAML scalar literal |
| `overlay.py` / `overlay.yaml` | hand-authored type / items / enum / range / backend / advanced annotations literals can't express |
| `gates.py` | translate `recycling_compatibility.yaml` into compat rules + field-visibility gates |
| `emit.py` | assemble a per-surface JSON Schema (folds all of the above) |
| `generate.py` | serialize the schemas to `website/src/generated/` (JSON-safe) |

The four surfaces are `data`, `pretrain`, `distill` (from comment-rich `get-yaml` templates) and
`eval` (from the `EvalConfig` dataclass). `data` / `pretrain` templates live as packaged YAML under
`src/nanoplm/cli/templates/`; `distill` is still an embedded string in `distill.get_yaml`.
`nanoplm eval get-yaml` also ships a packaged `eval.yaml`, but its template curates a subset — it
comments out optional, auto-defaulted fields (`device`, `embedder_name`, `tasks`) — so the dataclass
is used as eval's *docs* source, because it documents *every* settable field with its per-field
comments.

## The parity gate (the keystone)

`tests/docs_schema/test_parity.py` fails if **any** user-facing config field is undocumented, or
if **any** documented field doesn't exist on the real dataclass. This is what keeps the docs from
ever drifting from the code. Fields that exist on a dataclass but are auto-populated (e.g. from the
`.data_manifest`) and never set by the user are listed in `PARITY_INTERNAL_ALLOWLIST` in
`sources.py`, with comments explaining why they are excluded.

## Notes

- Non-finite float defaults (e.g. `max_grad_norm: .inf`) are serialized as the JSON-safe **string**
  sentinel `".inf"` (JSON has no `Infinity`; the string mirrors the YAML literal users type). So in
  the artifacts a `"type": "number"` field may carry a string `".inf"` default — expected, not a bug.
- A template field whose default is `null` has no type to infer, so generation fails with
  `<section>.<field> defaults to null` until `overlay.yaml` gives it a nullable union, e.g.
  `type: [integer, "null"]` (plus `items:` for a list). The union is emitted as a JSON type array.
- The generated `website/src/generated/*.schema.json` files are committed. The `docs` CI workflow
  regenerates them and a drift check (`git diff --exit-code`) fails the build if they're stale — so
  run the generator and commit the result after any template/dataclass/overlay change.
