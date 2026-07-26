const authors = [
  { name: "Dao Sy Duy Minh", equal: true },
  { name: "Huynh Trung Kiet", equal: true },
  { name: "Chi-Nguyen Tran", equal: false },
  { name: "Nguyen Lam Phu Quy", equal: false },
  { name: "Phu-Hoa Pham", equal: false },
];

const threats = [
  {
    code: "T1",
    tier: "Reliability",
    title: "Prompt and format",
    text: "A reasonable wording or response format changes the measured result.",
  },
  {
    code: "T2",
    tier: "Reliability",
    title: "Order and presentation",
    text: "Position, label, or presentation order changes a choice or verdict.",
  },
  {
    code: "T3",
    tier: "Validity",
    title: "Construct",
    text: "The probe rewards a shortcut or response style instead of the intended construct.",
  },
  {
    code: "T4",
    tier: "Validity",
    title: "Scoring and benchmark",
    text: "The evaluator or benchmark definition changes what counts as evidence.",
  },
  {
    code: "T5",
    tier: "Generalizability",
    title: "Population",
    text: "A selected set of models is treated as a human or deployment population.",
  },
  {
    code: "T6",
    tier: "Generalizability",
    title: "Ecological transfer",
    text: "A questionnaire or benchmark result is assumed to transfer to open behavior.",
  },
];

const bibtex = `@article{daosy2026believe,
  title   = {Can We Believe What Large Language Models Do?
             A Survey of Validity Threats in Behavioral Studies},
  author  = {Dao Sy, Duy Minh and Huynh, Trung Kiet and
             Tran, Chi-Nguyen and Nguyen, Lam Phu Quy and Pham, Phu-Hoa},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/technoob05/llm-behavior-validity-survey}
}`;

function Icon({ name }: { name: "paper" | "code" | "book" | "arrow" }) {
  const paths = {
    paper: (
      <>
        <path d="M6 2h8l4 4v16H6z" />
        <path d="M14 2v5h5M9 12h6M9 16h6" />
      </>
    ),
    code: (
      <>
        <path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" />
      </>
    ),
    book: (
      <>
        <path d="M4 5.5A3.5 3.5 0 0 1 7.5 2H11v18H7.5A3.5 3.5 0 0 0 4 23z" />
        <path d="M20 5.5A3.5 3.5 0 0 0 16.5 2H13v18h3.5A3.5 3.5 0 0 1 20 23z" />
      </>
    ),
    arrow: <path d="M5 12h14M14 7l5 5-5 5" />,
  };
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

export default function Home() {
  return (
    <main>
      <nav className="nav">
        <a className="brand" href="#top" aria-label="Back to top">
          <span className="brandMark">φ</span>
          <span>LLM Behavior Validity</span>
        </a>
        <div className="navLinks">
          <a href="#audit">Audit</a>
          <a href="#evidence">Evidence</a>
          <a href="#resources">Resources</a>
          <a href="#citation">Citation</a>
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="aurora one" />
        <div className="aurora two" />
        <div className="heroInner">
          <p className="eyebrow">Survey · Behavioral evaluation · 2026</p>
          <h1>
            Can We Believe What
            <span> Large Language Models Do?</span>
          </h1>
          <p className="subtitle">
            A Survey of Validity Threats in Behavioral Studies
          </p>
          <div className="authors" aria-label="Authors">
            {authors.map((author, index) => (
              <span key={author.name}>
                {author.name}
                {author.equal && <sup>†</sup>}
                {index < authors.length - 1 && <i>·</i>}
              </span>
            ))}
          </div>
          <p className="affiliation">
            Faculty of Information Technology, University of Science, VNU-HCM
            <br />
            <small>† Equal contribution</small>
          </p>
          <div className="actions">
            <a className="button primary" href="paper.pdf">
              <Icon name="paper" /> Read the paper
            </a>
            <a
              className="button"
              href="https://github.com/technoob05/llm-behavior-validity-survey"
            >
              <Icon name="code" /> Explore the code
            </a>
            <a className="button" href="literature/">
              <Icon name="book" /> 281 surveyed works
            </a>
          </div>
        </div>
      </section>

      <section className="statement wrap">
        <p className="sectionLabel">The inference gap</p>
        <h2>A score can be correct while the claim built from it is wrong.</h2>
        <p>
          Researchers use model outputs to infer traits, preferences, strategies,
          values, and social behavior. Yet an irrelevant change to wording,
          option order, scoring, or context can change the conclusion. This
          survey asks what must remain stable before one tested response supports
          a broader claim about a model.
        </p>
        <div className="claimFlow">
          <div>
            <strong>Tested response</strong>
            <span>one probe, order, and scorer</span>
          </div>
          <span className="flowArrow">→</span>
          <div className="danger">
            <strong>Inference gap</strong>
            <span>six places the claim can fail</span>
          </div>
          <span className="flowArrow">→</span>
          <div className="success">
            <strong>Bounded claim</strong>
            <span>model, probe pool, score, and scope</span>
          </div>
        </div>
      </section>

      <section className="numbersBand" aria-label="Survey highlights">
        <div className="numbers wrap">
          <article>
            <b>281</b>
            <span>unique cited works</span>
          </article>
          <article>
            <b>3</b>
            <span>audit tiers</span>
          </article>
          <article>
            <b>6</b>
            <span>recurring threats</span>
          </article>
          <article>
            <b>3</b>
            <span>public-data reanalyses</span>
          </article>
        </div>
      </section>

      <section className="audit wrap" id="audit">
        <div className="sectionHead">
          <div>
            <p className="sectionLabel">A claim-centered audit</p>
            <h2>Check the inference in the order failures propagate.</h2>
          </div>
          <p>
            Reliability comes first. A score that cannot be reproduced cannot
            establish construct validity, and a weak construct cannot support a
            transfer claim.
          </p>
        </div>
        <div className="tiers">
          <article className="tier reliability">
            <div className="tierIndex">01</div>
            <div>
              <span>Reliability</span>
              <h3>Would the result repeat?</h3>
              <p>Vary prompts, formats, order, and independent runs.</p>
            </div>
          </article>
          <article className="tier validity">
            <div className="tierIndex">02</div>
            <div>
              <span>Validity</span>
              <h3>Does the score mean what we say?</h3>
              <p>Test construct alignment, shortcuts, scoring, and benchmarks.</p>
            </div>
          </article>
          <article className="tier general">
            <div className="tierIndex">03</div>
            <div>
              <span>Generalizability</span>
              <h3>Where does the claim travel?</h3>
              <p>Bound the model population, language, modality, and deployment.</p>
            </div>
          </article>
        </div>
      </section>

      <section className="threatSection" id="threats">
        <div className="wrap">
          <p className="sectionLabel">Six recurring threats</p>
          <h2>The taxonomy names the failure before choosing the remedy.</h2>
          <div className="threatGrid">
            {threats.map((threat) => (
              <article key={threat.code} className="threatCard">
                <div>
                  <span className="threatCode">{threat.code}</span>
                  <span className="threatTier">{threat.tier}</span>
                </div>
                <h3>{threat.title}</h3>
                <p>{threat.text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="evidence wrap" id="evidence">
        <div className="sectionHead">
          <div>
            <p className="sectionLabel">Evidence, not a universal ranking</p>
            <h2>Fragility belongs to a claim, probe axis, and model.</h2>
          </div>
          <p>
            The fragility fraction, φ, supplies a reference point: it reports how
            much measured variation is associated with one declared change to
            the test.
          </p>
        </div>
        <div className="evidenceGrid">
          <article className="largeFinding">
            <span className="findingKicker">Matched item-level analysis</span>
            <b>11×</b>
            <h3>larger strict fragility on the tested dilemma axis</h3>
            <p>
              This is a narrow contrast between two datasets and two probe axes.
              It is not evidence that behavior is generally more fragile than
              capability.
            </p>
          </article>
          <div className="smallFindings">
            <article>
              <b>75%</b>
              <h3>of model pairs changed order</h3>
              <p>somewhere across the tested MoralChoice variants.</p>
            </article>
            <article>
              <b>94 pp</b>
              <h3>largest observed preference swing</h3>
              <p>a single-setting extreme, not a typical effect.</p>
            </article>
            <article className="principle">
              <span>Core reporting rule</span>
              <p>
                Replace “model M has X” with a claim that names the model, probe
                pool, scoring rule, and observed fragility.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="figureSection">
        <div className="wrap figureWrap">
          <div>
            <p className="sectionLabel">The paper in one figure</p>
            <h2>From a single-probe shortcut to an auditable claim.</h2>
            <p>
              The framework connects psychometrics, machine psychology, agent
              evaluation, social simulation, and LLM-as-a-judge research through
              the same inference problem.
            </p>
          </div>
          <img
            src="hero.png"
            alt="The paper's claim-centered three-tier audit framework"
          />
        </div>
      </section>

      <section className="resources wrap" id="resources">
        <p className="sectionLabel">Open research resources</p>
        <h2>Read, reproduce, inspect, and extend.</h2>
        <div className="resourceGrid">
          <a href="paper.pdf">
            <span className="resourceIcon"><Icon name="paper" /></span>
            <div><h3>Paper</h3><p>Full 54-page manuscript and appendix.</p></div>
            <Icon name="arrow" />
          </a>
          <a href="literature/">
            <span className="resourceIcon"><Icon name="book" /></span>
            <div><h3>Literature catalogue</h3><p>Every one of the 281 cited works.</p></div>
            <Icon name="arrow" />
          </a>
          <a href="https://github.com/technoob05/llm-behavior-validity-survey/tree/main/analysis">
            <span className="resourceIcon"><Icon name="code" /></span>
            <div><h3>Reanalyses</h3><p>CPU-only scripts and numeric regression checks.</p></div>
            <Icon name="arrow" />
          </a>
          <a href="https://github.com/technoob05/llm-behavior-validity-survey/tree/main/artifact">
            <span className="resourceIcon"><Icon name="code" /></span>
            <div><h3>Artifact</h3><p>Provenance, governance, and pinned public sources.</p></div>
            <Icon name="arrow" />
          </a>
        </div>
      </section>

      <section className="citation" id="citation">
        <div className="wrap citationInner">
          <div>
            <p className="sectionLabel">Citation</p>
            <h2>Use the paper citation when building on this work.</h2>
            <p>
              The citation will be updated with the arXiv identifier and venue
              record when they become available.
            </p>
          </div>
          <pre><code>{bibtex}</code></pre>
        </div>
      </section>

      <footer>
        <div className="wrap footerInner">
          <div>
            <span className="brandMark">φ</span>
            <p>
              Can We Believe What Large Language Models Do?
              <br />
              <small>Built for transparent, claim-centered evaluation.</small>
            </p>
          </div>
          <div>
            <a href="paper.pdf">Paper</a>
            <a href="https://github.com/technoob05/llm-behavior-validity-survey">GitHub</a>
            <a href="#top">Back to top ↑</a>
          </div>
        </div>
      </footer>
    </main>
  );
}
