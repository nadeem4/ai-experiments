import type { NextConfig } from "next";

// The site is fully client-side, so it builds to static files in out/ and can be
// hosted anywhere. trailingSlash turns /rerank into rerank/index.html.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
};

export default nextConfig;
