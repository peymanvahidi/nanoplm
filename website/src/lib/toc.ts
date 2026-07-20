// website/src/lib/toc.ts
// Splices the config reference's field blocks into Starlight's table of contents.
//
// Starlight builds its ToC from the Markdown AST, so it only sees headings written
// as `##` in the .mdx. The block headings inside a config section come from the
// generated schema and are rendered by ReferenceFields.astro, so remark never sees
// them and they are missing from the right-hand panel, which is where a 160-field
// page most needs an index. This adds them back, under the section they belong to.
//
// The ids come from schema.ts's groupSlug(), the same helper the renderer uses, so
// the links resolve and Starlight's scroll-spy (which matches ToC hrefs against
// real heading ids in the DOM) highlights them while scrolling.

import { toRows, toGroups, type Surface } from './schema';
import data from '../generated/nanoplm.data.schema.json';
import pretrain from '../generated/nanoplm.pretrain.schema.json';
import evalSchema from '../generated/nanoplm.eval.schema.json';

export interface TocItem {
  depth: number;
  slug: string;
  text: string;
  children: TocItem[];
}

const SCHEMAS: Record<string, Surface> = {
  data: data as Surface,
  pretrain: pretrain as Surface,
  eval: evalSchema as Surface,
};

/** Which reference page renders which surface, and under which heading slugs. */
const PAGES: Record<string, { surface: string; sections: string[] }> = {
  'reference/pretrain': { surface: 'pretrain', sections: ['model', 'pretraining', 'resume'] },
  'reference/data': { surface: 'data', sections: ['data_dirs', 'data_params'] },
};

/** Depth-first search for the entry whose heading slug is `slug`. */
function find(items: TocItem[], slug: string): TocItem | undefined {
  for (const item of items) {
    if (item.slug === slug) return item;
    const hit = find(item.children, slug);
    if (hit) return hit;
  }
  return undefined;
}

/**
 * Return `items` with each config section's field blocks added as children.
 *
 * A no-op for any page not in PAGES, and for any section whose template carries
 * no `# ── Title ───` banners (data and eval have none today, so they keep the
 * flat ToC they have now). Never mutates the input.
 */
export function withFieldGroups(items: TocItem[], slug: string): TocItem[] {
  const page = PAGES[slug];
  if (!page) return items;
  const surface = SCHEMAS[page.surface];
  if (!surface) return items;

  const clone = structuredClone(items) as TocItem[];
  for (const section of page.sections) {
    const parent = find(clone, section);
    if (!parent) continue;
    const groups = toGroups(toRows(surface, section), surface, section).filter((g) => g.title);
    if (!groups.length) continue;
    parent.children = [
      ...parent.children,
      ...groups.map((g) => ({
        depth: parent.depth + 1,
        slug: g.slug,
        text: g.title,
        children: [],
      })),
    ];
  }
  return clone;
}
