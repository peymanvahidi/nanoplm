// website/tools/check-toc.mjs
// Post-build guard for the config reference's table of contents.
//
// The field group headings on the reference pages are rendered from the generated
// schema by ReferenceFields.astro, so remark never sees them and Starlight cannot
// index them on its own. src/routeData.ts splices them into the ToC instead.
//
// That patch rides on Starlight's route data API, which has changed shape before:
// the 0.30 to 0.41 upgrade moved route data off `Astro.props` and renamed `slug`
// to `id`, which silently emptied the patch. The build stayed green and the link
// check stayed green, because a ToC missing two thirds of its entries is still
// valid HTML pointing at real anchors. Only the rendered page knew, and nothing
// was reading it. Hence this check.
//
// The invariant carries no magic numbers and follows the schema: every field
// group heading rendered into the page body must also be linked from both the
// desktop and the mobile ToC.

import { readFileSync } from 'node:fs';

const PAGE = 'dist/reference/pretrain/index.html';
// Group heading ids come from schema.ts's groupSlug(), which spells them
// <surface>-<section>-group-<title>. The renderer and the ToC share that helper,
// so matching on `-group-` finds exactly the headings this patch is responsible
// for and stays correct as the schema gains and loses groups.
const GROUP_HEADING = /<h[1-6][^>]*\bid="([^"]*-group-[^"]*)"/g;
const TOCS = [
  ['desktop', /<starlight-toc[^>]*>[\s\S]*?<\/starlight-toc>/],
  ['mobile', /<mobile-starlight-toc[^>]*>[\s\S]*?<\/mobile-starlight-toc>/],
];

const html = readFileSync(PAGE, 'utf8');
const groups = [...new Set([...html.matchAll(GROUP_HEADING)].map((m) => m[1]))];

const problems = [];

// A page that renders no groups at all means the reference stopped rendering,
// which would otherwise make every containment check below vacuously true.
if (groups.length === 0) {
  problems.push(`${PAGE} rendered no field group headings at all.`);
}

for (const [name, pattern] of TOCS) {
  const region = html.match(pattern)?.[0];
  if (!region) {
    problems.push(`${name} ToC not found in ${PAGE}.`);
    continue;
  }
  const linked = new Set([...region.matchAll(/href="#([^"]+)"/g)].map((m) => m[1]));
  const missing = groups.filter((id) => !linked.has(id));
  if (missing.length) {
    problems.push(
      `${name} ToC is missing ${missing.length}/${groups.length} field group headings, ` +
        `starting with: ${missing.slice(0, 3).join(', ')}`
    );
  }
}

if (problems.length) {
  console.error('Config reference ToC check FAILED:');
  for (const p of problems) console.error(`  - ${p}`);
  console.error('\nThe field groups are spliced in by website/src/routeData.ts.');
  console.error("Starlight's route data API most likely changed shape again.");
  process.exit(1);
}

console.log(`ToC check passed: all ${groups.length} field group headings are linked from both ToCs.`);
