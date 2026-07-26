import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Can We Believe What Large Language Models Do?",
  description:
    "A survey of validity threats in behavioral studies of large language models.",
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
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
