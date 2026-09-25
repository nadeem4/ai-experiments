import type { Metadata } from "next";
import { IBM_Plex_Mono, Literata } from "next/font/google";
import { SiteFooter } from "@/components/site-footer";
import { SiteNav } from "@/components/site-nav";
import "./globals.css";

// Two faces with two jobs. Literata is a screen serif built for long reading and
// carries the prose; IBM Plex Mono has true tabular figures, so every number,
// axis label and doc id lines up in a column. The Decision Arena's highway
// lettering is deliberately not here: that site shows driving games, this one
// shows a retrieval benchmark, and the language should not be borrowed.
// Literata is variable, so asking for no particular weight ships one file per
// style instead of one per weight.
const literata = Literata({
  variable: "--font-literata",
  subsets: ["latin"],
  style: ["normal", "italic"],
});
const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

const DESCRIPTION =
  "Measured experiments on AI models, each one carrying the raw wire it was measured from: what a typed decision costs at 4 options and at 151, whether a token budget explains Laya's Banking77 score, and whether a decision model re-ranks retrieval better than BM25.";

export const metadata: Metadata = {
  metadataBase: new URL("https://lab.codewithnk.com"),
  title: "AI Experiments",
  description: DESCRIPTION,
  openGraph: {
    type: "website",
    siteName: "AI Experiments",
    url: "https://lab.codewithnk.com",
    title: "AI Experiments",
    description: DESCRIPTION,
  },
  twitter: { card: "summary_large_image", title: "AI Experiments", description: DESCRIPTION },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${literata.variable} ${plexMono.variable} antialiased`}>
      <body className="flex min-h-[100dvh] flex-col overflow-x-hidden">
        <SiteNav />
        <div className="flex-1">{children}</div>
        <SiteFooter />
      </body>
    </html>
  );
}
