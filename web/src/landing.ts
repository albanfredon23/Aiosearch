import { type BenchDataset, type BenchReport, type BenchValue, loadBenchmarks } from "./api";

const HIGHLIGHTS: { dataset: string; metric: string; title: string }[] = [
  { dataset: "hotpotqa", metric: "sp_all@2", title: "HotpotQA : les 2 paragraphes utiles trouvés (top 2)" },
  { dataset: "hotpotqa", metric: "sp_recall@2", title: "HotpotQA : paragraphes utiles retrouvés (top 2)" },
  { dataset: "hotpotqa", metric: "answer_in_top", title: "HotpotQA : bonne réponse affichée en tête, sans LLM" },
  { dataset: "fever", metric: "evidence_recall@1", title: "FEVER : preuve exacte en 1re position" },
];

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const pct = (v: number): string => `${(v * 100).toFixed(1)} %`;
const sentence = (text: string): string => text.charAt(0).toUpperCase() + text.slice(1);

function estimate(v: BenchValue | { mean: number; low: number; high: number } | undefined, signed = false): HTMLElement {
  const cell = el("td");
  if (!v) {
    cell.textContent = "—";
    return cell;
  }
  const sign = signed && v.mean > 0 ? "+" : "";
  cell.append(el("strong", undefined, `${sign}${(v.mean * 100).toFixed(1)}`));
  cell.append(el("small", undefined, ` [${(v.low * 100).toFixed(1)} ; ${(v.high * 100).toFixed(1)}]`));
  if (signed) cell.dataset.sign = v.low > 0 ? "up" : v.high < 0 ? "down" : "flat";
  return cell;
}

function qualityTable(data: BenchDataset, systems: string[], labels: Record<string, string>): HTMLTableElement {
  const table = el("table");
  table.append(el("caption", undefined, "Qualité, en % (IC 95 % par bootstrap apparié)"));
  const head = el("tr");
  head.append(el("th", undefined, "Mesure"), ...systems.map((s) => el("th", undefined, labels[s] ?? s)),
    el("th", undefined, "Écart AIOTECH − RAG"));
  table.append(el("thead"));
  table.tHead?.append(head);
  const body = el("tbody");
  for (const metric of data.metrics) {
    const row = el("tr");
    row.append(el("th", undefined, metric.label));
    for (const s of systems) row.append(estimate(metric.values[s]?.n ? metric.values[s] : undefined));
    row.append(estimate(metric.difference_vs_rag?.aiotech, true));
    body.append(row);
  }
  table.append(body);
  return table;
}

function costTable(data: BenchDataset, systems: string[], labels: Record<string, string>): HTMLTableElement {
  const table = el("table");
  table.append(el("caption", undefined, "Coût de calcul par requête"));
  const head = el("tr");
  head.append(el("th", undefined, "Mesure"), ...systems.map((s) => el("th", undefined, labels[s] ?? s)));
  table.append(el("thead"));
  table.tHead?.append(head);
  const rows: [string, (s: string) => string][] = [
    ["Latence p50", (s) => `${data.efficiency[s].latency_ms.p50.toFixed(1)} ms`],
    ["Latence p95", (s) => `${data.efficiency[s].latency_ms.p95.toFixed(1)} ms`],
    ["Temps CPU", (s) => `${data.efficiency[s].cpu_ms_per_query.toFixed(1)} ms`],
    ["Énergie", (s) => `${data.efficiency[s].energy_per_query.value_mwh.toFixed(4)} mWh`],
    ["Pic mémoire Python", (s) => `${data.efficiency[s].python_peak_mb ?? "—"} Mo`],
    ["VRAM", () => "0 Mo"],
    ["Jetons LLM", (s) => `${data.efficiency[s].tokens_in_per_query + data.efficiency[s].tokens_out_per_query}`],
    ["Coût API", (s) => {
      const c = data.efficiency[s].cost_usd_per_query;
      return c === null ? "inconnu" : `${c.toFixed(4)} $`;
    }],
  ];
  const body = el("tbody");
  for (const [label, render] of rows) {
    const row = el("tr");
    row.append(el("th", undefined, label), ...systems.map((s) => el("td", undefined, render(s))));
    body.append(row);
  }
  table.append(body);
  return table;
}

function tiles(report: BenchReport): HTMLElement {
  const grid = el("div", "tiles");
  for (const h of HIGHLIGHTS) {
    const metric = report.datasets[h.dataset]?.metrics.find((m) => m.metric === h.metric);
    const aio = metric?.values.aiotech;
    const rag = metric?.values.rag;
    if (!metric || !aio || !rag) continue;
    const tile = el("div", "tile");
    tile.append(el("span", "tile-title", h.title), el("strong", undefined, pct(aio.mean)),
      el("span", "tile-sub", `RAG classique : ${pct(rag.mean)} · n = ${aio.n}`));
    grid.append(tile);
  }
  const hotpot = report.datasets.hotpotqa?.efficiency.aiotech;
  if (hotpot) {
    const tile = el("div", "tile");
    tile.append(el("span", "tile-title", "Coût d'une recherche vérifiée (CPU)"),
      el("strong", undefined, `${hotpot.latency_ms.p50.toFixed(0)} ms`),
      el("span", "tile-sub", `p95 ${hotpot.latency_ms.p95.toFixed(0)} ms · 0 Mo de VRAM · ${hotpot.energy_per_query.value_mwh.toFixed(3)} mWh`));
    grid.append(tile);
  }
  return grid;
}

export async function renderBenchmarks(container: HTMLElement): Promise<void> {
  const report = await loadBenchmarks();
  container.replaceChildren();
  if (!report || Object.keys(report.datasets).length === 0) {
    container.append(el("p", "muted", "Les mesures seront publiées ici après exécution du benchmark : docker compose run --rm bench"));
    return;
  }
  const labels = report.systems;
  container.append(
    el("p", "bench-meta", `Mesuré le ${report.generated_at} sur ${report.host.platform}, ${report.host.cpu_count} cœurs, `
      + `CPU uniquement, sans GPU. Lecteur LLM : ${report.llm.status}.`),
    tiles(report),
  );
  for (const data of Object.values(report.datasets)) {
    const systems = Object.keys(data.efficiency);
    const block = el("article", "bench-dataset");
    block.append(el("h3", undefined, `${data.dataset} · n = ${data.n}`), el("p", "muted", `${sentence(data.sampling)}. ${sentence(data.setting)}.`));
    const scroll = el("div", "table-scroll");
    scroll.append(qualityTable(data, systems, labels));
    const scroll2 = el("div", "table-scroll");
    scroll2.append(costTable(data, systems, labels));
    block.append(scroll, scroll2);
    const energy = Object.values(data.efficiency)[0]?.energy_per_query.method;
    if (energy) block.append(el("p", "muted small", `Énergie : ${energy}. FLOPs d'un LLM distant : non mesurables côté client.`));
    container.append(block);
  }
}
