# Templates — comment style

These YAML files are user-facing templates copied by `nanoplm <cmd> get-yaml`. They are a config surface, not documentation.

## Rules

- Do not use complex sentences and words.
- Keep comments short. One line per field, ideally as a trailing `#` comment.
- A feature block gets **at most 2 lines** of preamble: what it does in one clause in a way that is easy for humans to understand.
- No mechanism explanations, no derivations, no ASCII diagrams, no restating what the field name already says.
- Long-form docs belong on the documentation website (`website/src/content/docs/`). There we can have the paper link and more information about which each parameter is.
- When you are adding a new feature from a research paper use the default value of them.
- When adding a feature, prefer extending an existing `── Section ──` over creating a new one.

Adding a field should add ~1 line, not a paragraph.
