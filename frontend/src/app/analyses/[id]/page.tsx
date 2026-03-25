import AnalysisWorkspacePage from "./client";

export async function generateStaticParams() {
  // Generate a placeholder so Next.js outputs an HTML shell.
  // CloudFront/Amplify rewrites unknown /analyses/* paths to this.
  return [{ id: "_" }];
}

export default function Page() {
  return <AnalysisWorkspacePage />;
}
