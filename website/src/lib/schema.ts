// website/src/lib/schema.ts
// Pure transform: a generated nanoplm.<surface>.schema.json -> flat reference rows
// for the documentation config reference. No runtime deps; consumed by ReferenceFields.astro.

export interface FieldSchema {
  type?: string | string[];
  default?: unknown;
  enum?: (string | number)[];
  description?: string;
  'x-nanoplm-backend'?: string[];
  'x-nanoplm-advanced'?: boolean;
  'x-nanoplm-visible-when'?: { field: string; op: string; value: unknown };
}

export interface SectionSchema {
  type: 'object';
  properties: Record<string, FieldSchema>;
}

export interface Surface {
  $schema: string;
  'x-nanoplm-surface': string;
  type: 'object';
  properties: Record<string, FieldSchema | SectionSchema>;
  required?: string[];
  'x-nanoplm-compat'?: unknown;
}

export interface Row {
  key: string;
  type: string;
  default?: unknown;
  enum?: (string | number)[];
  description?: string;
  backend?: string[];
  advanced?: boolean;
  visibleWhen?: { field: string; op: string; value: unknown };
  required?: boolean;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Rejoin the text either side of a removed run, tidying the punctuation that
 *  held it in place (a trailing `:`, a leading `,`, a sole parenthetical aside). */
function rejoin(before: string, after: string): string {
  const b = before
    .replace(/[\s:;,|/-]+$/, '')
    // The connector that introduced the list ("... . One of:") goes with it.
    .replace(/[\s(]*\b(?:one of|allowed|choices?|options?)$/i, '')
    .replace(/[\s:;,|/-]+$/, '')
    .trim();
  let a = after.replace(/^[\s:;,|]+/, '').trim();
  if (/^[.;,]$/.test(a)) a = ''; // the run ended the sentence
  const isAside = /^\(.*\)$/.test(a);
  if (!b && isAside) a = a.slice(1, -1).trim();
  return [b, a].filter(Boolean).join(' ').trim();
}

/**
 * Drop a description's restatement of its own enum values.
 *
 * Every enum field in the generated schema repeats its values in the description
 * ("wsd | warmup_stable | cosine | linear"), because the YAML template comment is
 * the only place the tool documents them. Rendered next to a real list of choices
 * that is the same information twice, so the run is removed here rather than
 * upstream — the generated schema stays a faithful copy of the tool.
 *
 * Deliberately conservative: the run must list *every* value, be joined by
 * separators alone, and sit at the start or end of the description. That leaves
 * per-value explanations intact ("token=current masking; span=BERT-style ..."),
 * which carry information the choice list cannot.
 */
export function stripEnumRestatement(
  description: string,
  values: (string | number)[],
): string {
  if (!description || values.length < 2) return description;
  // Longest-first so `muonh` is preferred over `muon`; \b so `decay` does not
  // match inside `decay_tokens`.
  const alts = values
    .map((v) => escapeRe(String(v)))
    .sort((a, b) => b.length - a.length)
    .join('|');
  const value = `\\b(?:${alts})\\b`;
  const run = new RegExp(`\\(?\\s*${value}(?:\\s*[|/,]\\s*${value})+\\s*\\)?`, 'g');
  const wanted = new Set(values.map(String));

  for (const m of description.matchAll(run)) {
    const listed = new Set(m[0].match(new RegExp(value, 'g')) ?? []);
    if (listed.size !== wanted.size) continue; // a partial list may be prose
    const before = description.slice(0, m.index).trim();
    const after = description.slice(m.index + m[0].length).trim();
    // Anchored at either end; a trailing parenthetical aside or a bare full stop
    // still counts as the end, so "TE only: a | b (c is Blackwell+ only)" keeps
    // its caveat and "... One of: a, b, c." loses the whole clause.
    const endish = after === '' || /^\(.*\)$/.test(after) || /^[.;,]$/.test(after);
    if (before !== '' && !endish) continue;
    return rejoin(before, after);
  }
  return description;
}

/**
 * Flatten one surface (optionally one section of it) into reference rows.
 * Template surfaces (data/pretrain/distill) are nested by section
 * (`properties[section].properties[field]`); the eval surface is flat
 * (`properties[field]`), so call `toRows(surface)` with no section for it.
 */
export function toRows(surface: Surface, section?: string): Row[] {
  const node: Record<string, FieldSchema> = section
    ? ((surface.properties[section] as SectionSchema)?.properties ?? {})
    : (surface.properties as Record<string, FieldSchema>);
  const required = new Set(surface.required ?? []);

  return Object.entries(node).map(([key, field]) => {
    const type = Array.isArray(field.type) ? field.type.join(' | ') : (field.type ?? 'string');
    const row: Row = { key, type };
    if ('default' in field) row.default = field.default;
    if (field.enum) row.enum = field.enum;
    if (field.description) {
      const desc = field.enum
        ? stripEnumRestatement(field.description, field.enum)
        : field.description;
      if (desc) row.description = desc;
    }
    if (field['x-nanoplm-backend']) row.backend = field['x-nanoplm-backend'];
    if (field['x-nanoplm-advanced']) row.advanced = true;
    if (field['x-nanoplm-visible-when']) row.visibleWhen = field['x-nanoplm-visible-when'];
    if (required.has(key)) row.required = true;
    return row;
  });
}
