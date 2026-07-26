"use client";

import { useMemo, useState } from "react";

export type Work = {
  year: string;
  title: string;
  url: string;
  authors: string;
  key: string;
};

export default function LiteratureExplorer({ papers }: { papers: Work[] }) {
  const [query, setQuery] = useState("");
  const [year, setYear] = useState("all");
  const years = useMemo(
    () => [...new Set(papers.map((paper) => paper.year))].sort().reverse(),
    [papers],
  );
  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return papers.filter((paper) => {
      const matchesYear = year === "all" || paper.year === year;
      const haystack =
        `${paper.title} ${paper.authors} ${paper.key}`.toLocaleLowerCase();
      return matchesYear && (!needle || haystack.includes(needle));
    });
  }, [papers, query, year]);

  return (
    <>
      <div className="catalogueTools" role="search">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search titles, authors, or citation keys"
          aria-label="Search surveyed literature"
        />
        <select
          value={year}
          onChange={(event) => setYear(event.target.value)}
          aria-label="Filter surveyed literature by year"
        >
          <option value="all">All years</option>
          {years.map((value) => (
            <option value={value} key={value}>{value}</option>
          ))}
        </select>
      </div>
      <div className="catalogueMeta" aria-live="polite">
        <span>{filtered.length} of {papers.length} cited works</span>
        <span>Sorted by year, newest first</span>
      </div>
      {filtered.map((paper) => {
        const originalIndex = papers.findIndex((item) => item.key === paper.key);
        return (
          <article className="work" key={paper.key}>
            <span className="workNumber">
              {String(originalIndex + 1).padStart(3, "0")}
            </span>
            <div>
              <div className="workMeta">
                <span>{paper.year}</span>
                <code>{paper.key}</code>
              </div>
              <h2>
                {paper.url ? (
                  <a href={paper.url} target="_blank" rel="noreferrer">
                    {paper.title}
                  </a>
                ) : paper.title}
              </h2>
              <p>{paper.authors}</p>
            </div>
          </article>
        );
      })}
      {filtered.length === 0 && (
        <p className="emptyCatalogue">
          No surveyed work matches this search. Try a broader title, author, or
          year.
        </p>
      )}
    </>
  );
}
