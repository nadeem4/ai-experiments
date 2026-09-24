import type { Metadata } from "next";
import { Overpass, Overpass_Mono } from "next/font/google";
import { SiteFooter } from "@/components/site-footer";
import { SiteNav } from "@/components/site-nav";
import "./globals.css";

// Overpass is an open-source take on Highway Gothic, the lettering on US road
// signs: the same face the Decision Arena uses, because this is its sibling.
const overpass = Overpass({ variable: "--font-overpass", subsets: ["latin"], weight: ["400", "600", "800"] });
const overpassMono = Overpass_Mono({ variable: "--font-overpass-mono", subsets: ["latin"], weight: ["400", "600"] });

const DESCRIPTION =
  "Measured experiments on AI models, each one carrying the raw wire it was measured from. First: does a decision model re-rank retrieval better than BM25?";

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
    <html lang="en" className={`${overpass.variable} ${overpassMono.variable} antialiased`}>
      <body className="flex min-h-[100dvh] flex-col overflow-x-hidden">
        <SiteNav />
        <div className="flex-1">{children}</div>
        <SiteFooter />
      </body>
    </html>
  );
}
