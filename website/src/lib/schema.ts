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
  'x-nanoplm-group'?: string;
  'x-nanoplm-visible-when'?: { field: string; op: string; value: unknown };
}

export interface SectionSchema {
  type: 'object';
  properties: Record<string, FieldSchema>;
  /** A name -> spec map (data's `sources`): the spec every entry follows. */
  additionalProperties?: SectionSchema;
  /** Banner-intro prose per group, keyed by group title. */
  'x-nanoplm-group-notes'?: Record<string, string>;
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
  group?: string;
  visibleWhen?: { field: string; op: string; value: unknown };
  required?: boolean;
}

/** One `# ── Title ───` block of a template section, in template order. */
export interface Group {
  title: string;
  slug: string;
  /** The template's own intro prose for the block, if it has any. */
  note?: string;
  rows: Row[];
  /**
   * Every field in the block carries `x-nanoplm-advanced`. The reference renders
   * all fields alike, so nothing reads this today; it is kept because it is the
   * one place the schema's per-field flag rolls up to a block.
   */
  advanced: boolean;
}

/**
 * The DOM id for a group heading.
 *
 * Shared by the renderer and the table-of-contents override. The ToC links to
 * these ids and Starlight's scroll-spy matches on them, so the two must agree
 * exactly. `_` is kept as a word character so `pure_torch` survives intact.
 */
export function groupSlug(prefix: string, title: string): string {
  const body = title
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `${prefix}-group-${body}`;
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
 * upstream, so the generated schema stays a faithful copy of the tool.
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

/** Whether a property is a block of fields rather than a field. */
function isSection(node: unknown): node is SectionSchema {
  const n = node as SectionSchema | undefined;
  return !!n && (n.properties !== undefined || n.additionalProperties !== undefined);
}

/**
 * The block of fields a section path names. A path is dotted for a nested block
 * (`sources.cluster_subsampling`), and a name -> spec map (`sources`) stands for
 * the spec its entries follow.
 */
export function sectionNode(surface: Surface, section: string): SectionSchema | undefined {
  let node: unknown = surface;
  for (const part of section.split('.')) {
    node = (node as SectionSchema | undefined)?.properties?.[part];
    const spec = (node as SectionSchema | undefined)?.additionalProperties;
    if (spec) node = spec;
  }
  return isSection(node) ? node : undefined;
}

/**
 * The DOM id prefix for a section's fields. A nested block is named by its last
 * segment (`data-cluster_subsampling-members`), so a link to a field keeps working
 * when its block moves under another one.
 */
export function sectionPrefix(surface: Surface, section?: string): string {
  const name = surface['x-nanoplm-surface'];
  return section ? `${name}-${section.split('.').pop()}` : name;
}

/**
 * Flatten one surface (optionally one section of it) into reference rows.
 * Template surfaces (data/pretrain/distill) are nested by section
 * (`properties[section].properties[field]`); see sectionNode() for nested paths.
 * With no section, the surface's top-level fields: all of eval's, or data's
 * `output_dir` (its sections are left to their own calls).
 */
export function toRows(surface: Surface, section?: string): Row[] {
  const node: Record<string, FieldSchema> = section
    ? (sectionNode(surface, section)?.properties ?? {})
    : Object.fromEntries(
        Object.entries(surface.properties).filter(([, f]) => !isSection(f)),
      ) as Record<string, FieldSchema>;
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
    if (field['x-nanoplm-group']) row.group = field['x-nanoplm-group'];
    if (field['x-nanoplm-visible-when']) row.visibleWhen = field['x-nanoplm-visible-when'];
    if (required.has(key)) row.required = true;
    return row;
  });
}

/**
 * Fold rows into the template's `# ── Title ───` blocks, preserving order.
 *
 * `toRows` already returns fields grouped and in template order (the generator
 * keeps each block contiguous), so this only has to break the run whenever the
 * group label changes. A section with no groups (eval, say) yields a single
 * untitled group, which renders exactly as the flat list did.
 */
export function toGroups(rows: Row[], surface: Surface, section?: string): Group[] {
  const prefix = sectionPrefix(surface, section);
  const notes = section ? (sectionNode(surface, section)?.['x-nanoplm-group-notes'] ?? {}) : {};
  const groups: Group[] = [];
  for (const row of rows) {
    const title = row.group ?? '';
    let last = groups[groups.length - 1];
    if (!last || last.title !== title) {
      last = { title, slug: title ? groupSlug(prefix, title) : '', rows: [], advanced: true };
      if (title && notes[title]) last.note = notes[title];
      groups.push(last);
    }
    last.rows.push(row);
    if (!row.advanced) last.advanced = false;
  }
  return groups;
}
