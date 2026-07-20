// website/src/routeData.ts
// Starlight route middleware. Splices the config reference's field blocks into
// the table of contents before anything renders (see lib/toc.ts for why remark
// never sees those headings).
//
// This is route middleware rather than a TableOfContents component override
// because the desktop and mobile ToC read the same `starlightRoute.toc`, and
// withFieldGroups appends unconditionally. Patching from both components would
// walk the same object twice and list every field group twice. Here it runs
// once per page render, and both ToCs pick the result up.

import { defineRouteMiddleware } from '@astrojs/starlight/route-data';
import { withFieldGroups, type TocItem } from './lib/toc';

export const onRequest = defineRouteMiddleware((context) => {
  const route = context.locals.starlightRoute;
  // Splash pages and any page without headings carry no toc. withFieldGroups is
  // itself a no-op for every id outside its PAGES map, so the rest of the site
  // passes through untouched.
  if (!route.toc) return;
  route.toc.items = withFieldGroups(route.toc.items as TocItem[], route.id);
});
