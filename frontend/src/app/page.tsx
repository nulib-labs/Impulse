"use client";

import "@/lib/amplify-config";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getCurrentUser, signInWithRedirect } from "aws-amplify/auth";

export default function Home() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    getCurrentUser()
      .then(() => {
        // Already signed in, go to dashboard
        router.replace("/dashboard");
      })
      .catch(() => {
        // Not signed in, show landing page
        setChecking(false);
      });
  }, [router]);

  if (checking) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "100vh",
        }}
      >
        Loading...
      </div>
    );
  }

  return (
    <main
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        gap: "1.5rem",
      }}
    >
      <h1 style={{ fontSize: "2rem", fontWeight: 700 }}>Impulse</h1>
      <p style={{ color: "var(--muted)" }}>
        Large-scale document processing and extraction
      </p>
      <button
        onClick={() => signInWithRedirect()}
        style={{
          padding: "0.75rem 2rem",
          backgroundColor: "var(--primary)",
          color: "white",
          border: "none",
          borderRadius: "0.375rem",
          fontSize: "1rem",
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        Sign In
      </button>
    </main>
  );
}
