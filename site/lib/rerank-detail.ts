import type { Detail } from "@/lib/rerank";

const DETAIL_URL = "/rerank-detail.json";
let pending: Promise<Detail> | null = null;

/**
 * The 1.4MB candidate file, fetched once however many sections ask for it. A
 * failed fetch is forgotten, so the next section to ask tries again.
 */
export function loadDetail(): Promise<Detail> {
  if (!pending) {
    pending = fetch(DETAIL_URL).then((response) =>
      response.ok ? (response.json() as Promise<Detail>) : Promise.reject(new Error("unavailable")),
    );
    pending.catch(() => {
      pending = null;
    });
  }
  return pending;
}
