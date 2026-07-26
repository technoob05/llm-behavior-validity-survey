import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(
    "https://technoob05.github.io/llm-behavior-validity-survey/",
  ),
  title: "Can We Believe What Large Language Models Do?",
  description:
    "A survey of validity threats in behavioral studies of large language models.",
  authors: [
    { name: "Dao Sy Duy Minh" },
    { name: "Huynh Trung Kiet" },
    { name: "Chi-Nguyen Tran" },
    { name: "Nguyen Lam Phu Quy" },
    { name: "Phu-Hoa Pham" },
  ],
  keywords: [
    "large language models",
    "behavioral evaluation",
    "validity",
    "reliability",
    "generalizability",
    "LLM-as-a-judge",
  ],
  openGraph: {
    title: "Can We Believe What Large Language Models Do?",
    description:
      "A claim-centered audit of reliability, validity, and generalizability.",
    type: "article",
    url: "https://technoob05.github.io/llm-behavior-validity-survey/",
    images: [
      {
        url: "hero.png",
        alt: "Claim-centered audit framework for behavioral studies of large language models",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Can We Believe What Large Language Models Do?",
    description:
      "A claim-centered audit of reliability, validity, and generalizability.",
    images: ["hero.png"],
  },
  alternates: {
    canonical: "https://technoob05.github.io/llm-behavior-validity-survey/",
  },
  icons: {
    icon: "icon.svg",
  },
  robots: {
    index: true,
    follow: true,
  },
  other: {
    citation_title: "Can We Believe What Large Language Models Do? A Survey of Validity Threats in Behavioral Studies",
    citation_author: [
      "Dao Sy Duy Minh",
      "Huynh Trung Kiet",
      "Chi-Nguyen Tran",
      "Nguyen Lam Phu Quy",
      "Phu-Hoa Pham",
    ],
    citation_publication_date: "2026",
    citation_pdf_url:
      "https://technoob05.github.io/llm-behavior-validity-survey/paper.pdf",
  },
};

const scholarlyArticle = {
  "@context": "https://schema.org",
  "@type": "ScholarlyArticle",
  headline:
    "Can We Believe What Large Language Models Do? A Survey of Validity Threats in Behavioral Studies",
  datePublished: "2026",
  author: [
    "Dao Sy Duy Minh",
    "Huynh Trung Kiet",
    "Chi-Nguyen Tran",
    "Nguyen Lam Phu Quy",
    "Phu-Hoa Pham",
  ].map((name) => ({
    "@type": "Person",
    name,
  })),
  url: "https://technoob05.github.io/llm-behavior-validity-survey/",
  encoding: {
    "@type": "MediaObject",
    contentUrl:
      "https://technoob05.github.io/llm-behavior-validity-survey/paper.pdf",
    encodingFormat: "application/pdf",
  },
  keywords:
    "large language models, behavioral evaluation, validity, reliability, generalizability, LLM-as-a-judge",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        {children}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(scholarlyArticle) }}
        />
      </body>
    </html>
  );
}
