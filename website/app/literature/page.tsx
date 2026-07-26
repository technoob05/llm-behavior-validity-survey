import fs from "node:fs";
import path from "node:path";
import LiteratureExplorer, { type Work } from "./LiteratureExplorer";

function works(): Work[] {
  const source = fs.readFileSync(
    path.join(process.cwd(), "data", "surveyed-literature.md"),
    "utf8",
  );
  return source
    .split(/\r?\n/)
    .filter((line) => /^\|\s*\d{4}\s*\|/.test(line))
    .map((line) => {
      const cells = line.split("|").slice(1, -1).map((cell) => cell.trim());
      const link = cells[1].match(/^\[(.*)\]\((.*)\)$/);
      return {
        year: cells[0],
        title: link?.[1] ?? cells[1],
        url: link?.[2] ?? "",
        authors: cells[2],
        key: cells[3].replaceAll("`", ""),
      };
    });
}

export default function LiteraturePage() {
  const papers = works();
  return (
    <main className="literaturePage">
      <header className="literatureHero">
        <div className="wrap">
          <a className="backLink" href="../">← Project page</a>
          <p className="eyebrow">Complete evidence catalogue</p>
          <h1>281 works surveyed by the paper.</h1>
          <p>
            This list is generated directly from the manuscript bibliography.
            Every entry is cited in the paper, and every title links to its
            preferred public record when one is available.
          </p>
        </div>
      </header>
      <section className="literatureList wrap">
        <LiteratureExplorer papers={papers} />
      </section>
      <footer>
        <div className="wrap footerInner">
          <div>
            <span className="brandMark">φ</span>
            <p>LLM Behavior Validity Survey</p>
          </div>
          <div><a href="../">Project page</a><a href="../paper.pdf">Paper</a></div>
        </div>
      </footer>
    </main>
  );
}
