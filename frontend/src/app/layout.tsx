import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Impulse - Document Processing",
  description: "Large-scale document processing and extraction pipeline",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
