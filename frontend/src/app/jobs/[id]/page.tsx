import JobDetailPage from "./client";

export async function generateStaticParams() {
  return [{ id: "_" }];
}

export default function Page() {
  return <JobDetailPage />;
}
