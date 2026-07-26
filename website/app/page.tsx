const authors = [
  { name: "Dao Sy Duy Minh", equal: true },
  { name: "Huynh Trung Kiet", equal: true },
  { name: "Chi-Nguyen Tran", equal: false },
  { name: "Nguyen Lam Phu Quy", equal: false },
  { name: "Phu-Hoa Pham", equal: false },
];

const domains = [
  {
    number: "01",
    title: "Machine psychology",
    text: "Personality, values, attitudes, and dispositional claims, audited with psychometric tools.",
  },
  {
    number: "02",
    title: "Games and agents",
    text: "Choice, strategic reasoning, cooperation, and payoff-sensitive behavior.",
  },
  {
    number: "03",
    title: "Social simulation",
    text: "Synthetic populations, role-conditioned agents, and model societies.",
  },
  {
    number: "04",
    title: "LLM-as-a-judge",
    text: "Position bias, scorer reliability, calibration, and repeated judgments.",
  },
];

const threats = [
  {
    code: "T1",
    tier: "Reliability",
    title: "Prompt and format",
    text: "A reasonable wording or response format changes the measured result.",
    test: "Paraphrases, templates, response modes",
  },
  {
    code: "T2",
    tier: "Reliability",
    title: "Order and presentation",
    text: "Position, label, or presentation order changes a choice or verdict.",
    test: "Counterbalancing and order swaps",
  },
  {
    code: "T3",
    tier: "Validity",
    title: "Construct",
    text: "The probe rewards a shortcut or response style instead of the intended construct.",
    test: "Convergent and structural checks",
  },
  {
    code: "T4",
    tier: "Validity",
    title: "Scoring and benchmark",
    text: "The evaluator or benchmark definition changes what counts as evidence.",
    test: "Gold slices and scorer perturbations",
  },
  {
    code: "T5",
    tier: "Generalizability",
    title: "Population",
    text: "A selected set of models is treated as a human or deployment population.",
    test: "Model, role, and sampling boundaries",
  },
  {
    code: "T6",
    tier: "Generalizability",
    title: "Ecological transfer",
    text: "A questionnaire or benchmark result is assumed to transfer to open behavior.",
    test: "Languages, modalities, and deployment tasks",
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

const figureDimensions: Record<string, { width: number; height: number }> = {
  "figures/audit-framework.webp": { width: 2200, height: 1238 },
  "figures/fragility-forest.webp": { width: 1397, height: 768 },
  "figures/repeated-judges.webp": { width: 2200, height: 1467 },
  "figures/strict-vs-broad.webp": { width: 2200, height: 1467 },
};

function Icon({
  name,
}: {
  name: "paper" | "code" | "book" | "arrow" | "check" | "download";
}) {
  const paths = {
    paper: (
      <>
        <path d="M6 2h8l4 4v16H6z" />
        <path d="M14 2v5h5M9 12h6M9 16h6" />
      </>
    ),
    code: <path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" />,
    book: (
      <>
        <path d="M4 5.5A3.5 3.5 0 0 1 7.5 2H11v18H7.5A3.5 3.5 0 0 0 4 23z" />
        <path d="M20 5.5A3.5 3.5 0 0 0 16.5 2H13v18h3.5A3.5 3.5 0 0 1 20 23z" />
      </>
    ),
    arrow: <path d="M5 12h14M14 7l5 5-5 5" />,
    check: <path d="m5 12 4 4L19 6" />,
    download: (
      <>
        <path d="M12 3v12M7 10l5 5 5-5" />
        <path d="M5 21h14" />
      </>
    ),
  };
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

function Figure({
  src,
  alt,
  className = "",
}: {
  src: string;
  alt: string;
  className?: string;
}) {
  const dimensions = figureDimensions[src];
  return (
    <a
      className={`paperFigure ${className}`}
      href={src}
      target="_blank"
      rel="noreferrer"
      aria-label={`${alt}. Open full-size figure.`}
    >
      <img
        src={src}
        alt={alt}
        loading="lazy"
        width={dimensions?.width}
        height={dimensions?.height}
      />
      <span>Open full size <Icon name="arrow" /></span>
    </a>
  );
}

export default function Home() {
  return (
    <main id="main-content">
      <a className="skipLink" href="#main-content">Skip to main content</a>
      <nav className="nav">
        <a className="brand" href="#top" aria-label="Back to top">
          <span className="brandMark">φ</span>
          <span>LLM Behavior Validity</span>
        </a>
        <div className="navLinks">
          <a href="#framework">Framework</a>
          <a href="#evidence">Evidence</a>
          <a href="#practice">Practice</a>
          <a href="#coverage">Coverage</a>
          <a href="#resources">Resources</a>
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="aurora one" />
        <div className="aurora two" />
        <div className="heroInner">
          <div className="heroTag">
            <span>Survey</span>
            <span>Open artifact</span>
            <span>2026</span>
          </div>
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
              <Icon name="book" /> Search 281 works
            </a>
          </div>
          <div className="heroProof" aria-label="Paper summary">
            <div>
              <span>Core question</span>
              <strong>What must remain stable?</strong>
            </div>
            <div>
              <span>Audit result</span>
              <strong>Reject, narrow, or preserve</strong>
            </div>
            <div>
              <span>Reporting unit</span>
              <strong>Claim + probe axis + model</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="statement wrap">
        <p className="sectionLabel">The inference gap</p>
        <h2>A score can be correct while the claim built from it is wrong.</h2>
        <p>
          Behavioral studies use model outputs to infer traits, preferences,
          strategies, values, and social behavior. Yet a harmless change to
          wording, option order, scoring, or context can change the conclusion.
          The survey asks what evidence is needed before one tested response
          supports a broader claim about a model.
        </p>
        <div className="claimFlow">
          <div>
            <small>01</small>
            <strong>Tested response</strong>
            <span>One prompt, order, scorer, and setting</span>
          </div>
          <span className="flowArrow">→</span>
          <div className="danger">
            <small>02</small>
            <strong>Inference gap</strong>
            <span>Six recurring ways the conclusion can fail</span>
          </div>
          <span className="flowArrow">→</span>
          <div className="success">
            <small>03</small>
            <strong>Bounded claim</strong>
            <span>Model, probe pool, score, uncertainty, and scope</span>
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

      <section className="coverage wrap" id="coverage">
        <div className="sectionHead">
          <div>
            <p className="sectionLabel">One inference problem, four behavioral domains</p>
            <h2>The survey connects fields that usually use different language.</h2>
          </div>
          <p>
            The contribution is not another task catalogue. It traces the same
            inferential failure across behavioral domains, then gives each one a
            shared audit and reporting vocabulary.
          </p>
        </div>
        <div className="domainGrid">
          {domains.map((domain) => (
            <article key={domain.number}>
              <span>{domain.number}</span>
              <h3>{domain.title}</h3>
              <p>{domain.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="audit wrap" id="framework">
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
            <div className="tierTop">
              <div className="tierIndex">01</div>
              <span>Reliability</span>
            </div>
            <div>
              <h3>Would the result repeat?</h3>
              <p>Vary prompts, formats, order, and independent runs.</p>
              <ul>
                <li>Prompt and format dependence</li>
                <li>Order and presentation effects</li>
              </ul>
            </div>
          </article>
          <article className="tier validity">
            <div className="tierTop">
              <div className="tierIndex">02</div>
              <span>Validity</span>
            </div>
            <div>
              <h3>Does the score mean what we say?</h3>
              <p>Test construct alignment, shortcuts, scoring, and benchmarks.</p>
              <ul>
                <li>Construct and response-style mismatch</li>
                <li>Scoring and benchmark dependence</li>
              </ul>
            </div>
          </article>
          <article className="tier general">
            <div className="tierTop">
              <div className="tierIndex">03</div>
              <span>Generalizability</span>
            </div>
            <div>
              <h3>Where does the claim travel?</h3>
              <p>Bound the model population, language, modality, and deployment.</p>
              <ul>
                <li>Population and role assumptions</li>
                <li>Ecological and deployment transfer</li>
              </ul>
            </div>
          </article>
        </div>
      </section>

      <section className="visualFeature">
        <div className="wrap visualFeatureGrid">
          <div className="visualCopy">
            <p className="sectionLabel">The full audit, end to end</p>
            <h2>Start with the claim, not the benchmark.</h2>
            <p>
              Declare the behavioral claim and the harmless test detail that
              changes. Collect matched responses, measure the resulting
              variation, then reject, narrow, or preserve the conclusion.
            </p>
            <ol className="miniSteps">
              <li><b>Claim</b><span>Name exactly what is being inferred.</span></li>
              <li><b>Probe</b><span>Choose the test axis the claim should survive.</span></li>
              <li><b>Audit</b><span>Hold the model and items fixed where possible.</span></li>
              <li><b>Conclude</b><span>Report only the scope the evidence supports.</span></li>
            </ol>
          </div>
          <div>
            <Figure
              src="figures/audit-framework.webp"
              alt="Five-stage audit from a behavioral claim to a scoped conclusion"
            />
            <p className="figureCaption">
              Conceptual illustration. The example scores in the figure are not
              empirical results.
            </p>
          </div>
        </div>
      </section>

      <section className="threatSection" id="threats">
        <div className="wrap">
          <div className="sectionHead">
            <div>
              <p className="sectionLabel">Six recurring threats</p>
              <h2>Name the failure before choosing the remedy.</h2>
            </div>
            <p>
              The threats are ordered by the inference they attack. A mechanism
              can affect more than one tier, so the taxonomy is a diagnostic
              map, not six isolated boxes.
            </p>
          </div>
          <div className="threatGrid">
            {threats.map((threat) => (
              <article key={threat.code} className="threatCard">
                <div>
                  <span className="threatCode">{threat.code}</span>
                  <span className="threatTier">{threat.tier}</span>
                </div>
                <h3>{threat.title}</h3>
                <p>{threat.text}</p>
                <small>{threat.test}</small>
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
            The fragility fraction, φ, reports the share of measured variation
            associated with one declared change to the test. It is a design aid,
            not a universal pass-or-fail score.
          </p>
        </div>
        <div className="evidenceGrid">
          <article className="largeFinding">
            <span className="findingKicker">Matched item-level analysis</span>
            <b>≈21×</b>
            <h3>median gap across matched template draws</h3>
            <p>
              Across 200 matched template draws, the median ratio is 20.9, with
              a central 95% range from 10.9 to 68.6. The fixed illustration gives
              strict φ=0.0069 for MMLU and φ=0.1439 for the dilemmas. This is not
              a general comparison of capability and behavior.
            </p>
          </article>
          <div className="smallFindings">
            <article>
              <b>75%</b>
              <h3>of model pairs changed order</h3>
              <p>somewhere across the six tested MoralChoice variants.</p>
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
                pool, scoring rule, uncertainty, and observed fragility.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="chartSection">
        <div className="wrap chartGrid">
          <Figure
            src="figures/fragility-forest.webp"
            alt="Model-level fragility estimates and inversion-risk thresholds"
            className="chartFigure"
          />
          <div className="chartCopy">
            <p className="sectionLabel">A result becomes a design decision</p>
            <h2>More items do not replace more probe variants.</h2>
            <p>
              The forest plot places each model&apos;s estimated fragility beside
              conditional inversion-risk thresholds. If the research claim
              compares models, the relevant failure is a ranking that reverses
              when a reasonable test variant changes.
            </p>
            <div className="readingNotes">
              <div><b>Dots</b><span>behavioral estimates under the tested dilemmas</span></div>
              <div><b>Diamonds</b><span>broad estimates including item-by-probe interaction</span></div>
              <div><b>Green circles</b><span>clear-answer item estimates</span></div>
            </div>
            <p className="scopeNote">
              Read the thresholds conditionally: tolerated risk, model gap, and
              probe pool all matter.
            </p>
          </div>
        </div>
      </section>

      <section className="sensitivity wrap">
        <div className="sectionHead">
          <div>
            <p className="sectionLabel">Two complementary readings</p>
            <h2>Did the item change, and how important was that change?</h2>
          </div>
          <p>
            Strict and broad sensitivity answer different questions. Reporting
            both prevents an average that looks stable from hiding opposite
            item-level movements.
          </p>
        </div>
        <div className="sensitivityCards">
          <article>
            <span>Strict φ</span>
            <h3>Probe main effect</h3>
            <p>
              Measures whether the probe moves the overall score in a consistent
              direction across items.
            </p>
          </article>
          <article>
            <span>Broad φ</span>
            <h3>Main effect + interaction</h3>
            <p>
              Also counts cases where a probe helps some items and hurts others,
              even when the average barely moves.
            </p>
          </article>
          <article>
            <span>One observation per cell</span>
            <h3>Do not over-identify noise</h3>
            <p>
              Report strict φ and observed disagreement unless repeated
              observations separate interaction from run or judge noise.
            </p>
          </article>
        </div>
        <Figure
          src="figures/strict-vs-broad.webp"
          alt="Conceptual comparison of strict and broad sensitivity"
          className="wideFigure"
        />
        <p className="figureCaption centered">
          Conceptual illustration of the two estimands. See the paper and
          artifact for empirical values and uncertainty.
        </p>
      </section>

      <section className="outcomesSection">
        <div className="wrap">
          <p className="sectionLabel">What an audit can change</p>
          <h2>Not every sensitive result is false, and not every stable score supports a trait.</h2>
          <div className="outcomeGrid">
            <article className="reject">
              <span>Reject</span>
              <h3>Stable personality, as a broad claim</h3>
              <p>
                Equivalent prompts move scores, response style can explain model
                differences, and questionnaire results transfer poorly to
                open-ended behavior.
              </p>
            </article>
            <article className="narrow">
              <span>Narrow</span>
              <h3>Preference on the tested dilemmas</h3>
              <p>
                The evidence supports a model- and probe-indexed stability claim,
                not a context-free statement about a disposition.
              </p>
            </article>
            <article className="preserve">
              <span>Preserve</span>
              <h3>Judge position bias</h3>
              <p>
                The position effect survives the three-tier audit when stated
                with its tested scorer, order manipulation, and evaluation scope.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="practice wrap" id="practice">
        <div className="sectionHead">
          <div>
            <p className="sectionLabel">Match the remedy to the source</p>
            <h2>Diagnose first, then spend the evaluation budget.</h2>
          </div>
          <p>
            A mitigation acts on a specific source of variation. The strongest
            remedy in one corpus can be the weaker one in another.
          </p>
        </div>
        <div className="remedyGrid">
          <article>
            <div className="remedyMetric">88%</div>
            <h3>Order symmetrisation</h3>
            <p>
              Removed 88% of dilemma probe variance and 55% of pairwise verdict
              flips in the tested MoralChoice design.
            </p>
            <span>Targets position and order effects</span>
          </article>
          <article>
            <div className="remedyMetric">75%</div>
            <h3>Template ensembles</h3>
            <p>
              Removed 75% of probe variance on clear-answer items, where the
              ranking of the two remedies reversed.
            </p>
            <span>Targets wording and format dependence</span>
          </article>
          <article>
            <div className="remedyMetric">Gold</div>
            <h3>Gold-slice calibration</h3>
            <p>
              Anchors an automated scorer to trusted labels before the scorer is
              used to support a behavioral conclusion.
            </p>
            <span>Targets scoring and judge dependence</span>
          </article>
        </div>
      </section>

      <section className="judgeSection">
        <div className="wrap judgeGrid">
          <div className="visualCopy">
            <p className="sectionLabel">Reliability needs repeated evidence</p>
            <h2>One judge call cannot separate a prompt effect from judge noise.</h2>
            <p>
              Open-ended scoring adds a second stochastic system to the
              measurement. Repeated, independent judgments let real differences
              persist while random variation averages out.
            </p>
            <div className="checkList">
              <span><Icon name="check" /> Repeat independent judgments</span>
              <span><Icon name="check" /> Randomize answer position</span>
              <span><Icon name="check" /> Report consistency and signed bias</span>
              <span><Icon name="check" /> Keep a human-labeled calibration slice</span>
            </div>
          </div>
          <Figure
            src="figures/repeated-judges.webp"
            alt="Why repeated independent judge calls are needed"
          />
        </div>
      </section>

      <section className="checklist wrap">
        <div className="checklistIntro">
          <p className="sectionLabel">A compact protocol for a new study</p>
          <h2>Plan the claim and the audit together.</h2>
          <p>
            The paper&apos;s checklist turns validity from a post-hoc caveat into
            an experimental design. Each stage has a question, a stress test,
            and a reporting standard.
          </p>
          <a className="textLink" href="paper.pdf#page=35">
            Open the full checklist in the appendix <Icon name="arrow" />
          </a>
        </div>
        <div className="protocol">
          <article><b>01</b><div><h3>Bound the claim</h3><p>Name the model, construct, comparison, and intended scope.</p></div></article>
          <article><b>02</b><div><h3>Declare probe populations</h3><p>Predefine reasonable wording, order, format, scorer, or language variants.</p></div></article>
          <article><b>03</b><div><h3>Audit matched measurements</h3><p>Hold items and models fixed, estimate variation, and quantify uncertainty.</p></div></article>
          <article><b>04</b><div><h3>Test meaning and transfer</h3><p>Check construct evidence, scoring validity, and deployment-like behavior.</p></div></article>
          <article><b>05</b><div><h3>Report the surviving claim</h3><p>Reject, narrow, or preserve it with explicit limits and correction history.</p></div></article>
        </div>
      </section>

      <section className="figureSection">
        <div className="wrap figureWrap">
          <div>
            <p className="sectionLabel">The paper in one figure</p>
            <h2>From a single-probe shortcut to an auditable claim.</h2>
            <p>
              The framework uses psychometric tools to connect machine psychology,
              games and agents, social simulation, and LLM-as-a-judge research
              through the same inference problem.
            </p>
            <a className="textLink" href="paper.pdf">
              Read the complete argument <Icon name="arrow" />
            </a>
          </div>
          <img
            src="hero.png"
            alt="The paper's claim-centered three-tier audit framework"
            loading="lazy"
            width="1375"
            height="852"
          />
        </div>
      </section>

      <section className="resources wrap" id="resources">
        <p className="sectionLabel">Open research resources</p>
        <h2>Read, reproduce, inspect, and extend.</h2>
        <div className="resourceGrid">
          <a href="paper.pdf">
            <span className="resourceIcon"><Icon name="paper" /></span>
            <div><h3>Paper</h3><p>Full 54-page manuscript and navigable appendix.</p></div>
            <Icon name="arrow" />
          </a>
          <a href="literature/">
            <span className="resourceIcon"><Icon name="book" /></span>
            <div><h3>Searchable literature</h3><p>Every one of the 281 cited works.</p></div>
            <Icon name="arrow" />
          </a>
          <a href="https://github.com/technoob05/llm-behavior-validity-survey/tree/main/analysis">
            <span className="resourceIcon"><Icon name="code" /></span>
            <div><h3>Reanalyses</h3><p>CPU-only scripts and numeric regression checks.</p></div>
            <Icon name="arrow" />
          </a>
          <a href="https://github.com/technoob05/llm-behavior-validity-survey/tree/main/artifact">
            <span className="resourceIcon"><Icon name="download" /></span>
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
              The record will be updated with the arXiv identifier and venue
              metadata when they become available.
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
            <a href="literature/">Literature</a>
            <a href="https://github.com/technoob05/llm-behavior-validity-survey">GitHub</a>
            <a href="#top">Back to top ↑</a>
          </div>
        </div>
      </footer>
    </main>
  );
}
