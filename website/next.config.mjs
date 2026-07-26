const isGitHubPages = process.env.GITHUB_ACTIONS === "true";
const repository = "llm-behavior-validity-survey";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: isGitHubPages ? "export" : undefined,
  trailingSlash: true,
  images: { unoptimized: true },
  basePath: isGitHubPages ? `/${repository}` : "",
  assetPrefix: isGitHubPages ? `/${repository}/` : "",
};

export default nextConfig;
