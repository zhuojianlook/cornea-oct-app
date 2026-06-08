import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--c-bg)",
        surface: "var(--c-surface)",
        surface2: "var(--c-surface2)",
        border: "var(--c-border)",
        text: "var(--c-text)",
        "text-dim": "var(--c-text-dim)",
        accent: "var(--c-accent)",
        danger: "var(--c-red)",
        success: "var(--c-green)",
      },
    },
  },
  plugins: [],
} satisfies Config;
