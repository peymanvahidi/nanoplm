// website/src/lib/schema.ts
// Pure transform: a generated nanoplm.<surface>.schema.json -> flat reference rows
// for the documentation tables. No runtime deps; consumed by ReferenceTable.astro.

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
    if (field.description) row.description = field.description;
    if (field['x-nanoplm-backend']) row.backend = field['x-nanoplm-backend'];
    if (field['x-nanoplm-advanced']) row.advanced = true;
    if (field['x-nanoplm-visible-when']) row.visibleWhen = field['x-nanoplm-visible-when'];
    if (required.has(key)) row.required = true;
    return row;
  });
}
