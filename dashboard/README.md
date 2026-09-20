# Evaluation dashboard

React, TypeScript, and Vite dashboard for the FastAPI evaluation service. All displayed metrics and results come from `/api` responses; there is no seeded demo data.

```bash
cd dashboard
npm ci
npm run dev
```

The development server proxies `/api` to `http://localhost:8000`. Set `VITE_DEV_API_URL` to change that proxy target. For a separate production origin, set `VITE_API_BASE_URL` at build time; the default is same-origin `/api`. The included Dockerfile serves the static build through Nginx and proxies `/api` to the Compose service named `api`.

```bash
npm run build
```

The dataset importer accepts benchmark JSONL, a JSON array, or `{ "tasks": [...] }`. API keys and inference endpoints belong in server environment configuration, not browser form fields.
