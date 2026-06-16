/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkest: '#020617', // slate-950
        dark: '#0f172a',     // slate-900
        accent: '#38bdf8',    // sky-400
      },
    },
  },
  plugins: [],
}
