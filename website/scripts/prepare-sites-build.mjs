import { cpSync, existsSync, writeFileSync } from "node:fs";

if (!existsSync("dist/server/index.js") && !existsSync("dist/server/index.mjs")) {
  throw new Error("vinext did not emit a Sites-compatible server entrypoint");
}

if (!existsSync("dist/server/index.js")) {
  writeFileSync(
    "dist/server/index.js",
    'export { default } from "./index.mjs";\nexport * from "./index.mjs";\n',
    "utf8",
  );
}
cpSync(".openai", "dist/.openai", { recursive: true, force: true });
