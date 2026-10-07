export type Status = "FAIT" | "INFÉRENCE" | "INCERTAIN" | "NON VÉRIFIÉ";
export type Depth = "auto" | "light" | "standard" | "deep";

export interface Source {
  id: string;
  title: string;
  url: string | null;
  origin: "corpus" | "web";
  reliability: number;
}

export interface VerifiedClaim {
  text: string;
  status: Status;
  reason: string;
  sources: Source[];
  confidence: number;
}

export interface Possibility {
  rank: number;
  interpretation_id: string;
  interpretation_label: string;
  title: string;
  answer: string;
  reliability: number;
  reliability_pct: number;
  click_prior: number;
  claims: VerifiedClaim[];
  sources: Source[];
  alternatives: string[];
  synthesis: "template" | "llm";
}

export interface Neuron {
  id: string;
  label: string;
  facet: string;
  rationale: string;
  plausibility: number;
  subqueries: string[];
}

export interface Compute {
  tier: "light" | "standard" | "deep";
  complexity: number;
  max_interpretations: number;
  top_k: number;
  hops: number;
  effort: string;
}

export interface InterpretationsEvent {
  question_type: string;
  objective: string;
  planner: string;
  constraints: string[];
  entities: string[];
  compute: Compute;
  neurons: Neuron[];
}

export interface PassageRef {
  id: string;
  title: string;
  url: string | null;
  origin: "corpus" | "web";
  reliability: number;
  score: number;
}

export interface RetrievalEvent {
  interpretation_id: string;
  passages: PassageRef[];
  count?: number;
}

export interface ValueGroup {
  value_text: string;
  source_ids: string[];
  support: number;
}

export interface Contradiction {
  entity_label: string;
  attribute: string;
  groups: ValueGroup[];
  resolved: boolean;
  winner: string | null;
}

export interface GraphEvent {
  entities: number;
  claims: number;
  contradictions: Contradiction[];
}

export interface TrajectoryEvent {
  interpretation_id: string;
  admissible: boolean;
  rejection_reason: string | null;
  best: { label: string; reliability: number } | null;
  candidates: { label: string; rejected: boolean; reason: string | null; reliability: number }[];
}

export interface Usage {
  llm_calls: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number | null;
}

export interface SearchResult {
  search_id: string;
  query: string;
  blocked: boolean;
  block_reason: string | null;
  pii: { type: string; count: number }[];
  possibilities: Possibility[];
  contradictions: Contradiction[];
  warnings: string[];
  usage: Usage;
  metrics: Record<string, number | boolean | null>;
}

export interface StreamHandlers {
  start?: (data: { search_id: string; query: string; pii: { type: string; count: number }[] }) => void;
  blocked?: (data: { reason: string; tokens: number; cost_usd: number }) => void;
  interpretations?: (data: InterpretationsEvent) => void;
  retrieval?: (data: RetrievalEvent) => void;
  graph?: (data: GraphEvent) => void;
  trajectory?: (data: TrajectoryEvent) => void;
  possibilities?: (data: { possibilities: Possibility[] }) => void;
  warning?: (data: { message: string }) => void;
  done?: (data: { result: SearchResult }) => void;
  failure?: (message: string) => void;
}

const API = "/api/ui";

export function streamSearch(query: string, depth: Depth, web: boolean, handlers: StreamHandlers): () => void {
  const params = new URLSearchParams({ q: query, depth, web: String(web) });
  const source = new EventSource(`${API}/search/stream?${params.toString()}`);
  let finished = false;
  const finish = (): void => {
    finished = true;
    source.close();
  };
  const on = <T>(name: string, handler: ((data: T) => void) | undefined): void => {
    source.addEventListener(name, (event) => {
      if (!handler) return;
      try {
        handler(JSON.parse((event as MessageEvent<string>).data) as T);
      } catch (error) {
        console.error(`événement ${name} illisible`, error);
      }
    });
  };
  on("start", handlers.start);
  on("blocked", handlers.blocked);
  on("interpretations", handlers.interpretations);
  on("retrieval", handlers.retrieval);
  on("graph", handlers.graph);
  on("trajectory", handlers.trajectory);
  on("possibilities", handlers.possibilities);
  on("warning", handlers.warning);
  on<{ result: SearchResult }>("done", (data) => {
    finish();
    handlers.done?.(data);
  });
  on<{ message: string }>("error", (data) => {
    finish();
    handlers.failure?.(data.message);
  });
  source.onerror = () => {
    if (finished) return;
    finish();
    handlers.failure?.("Connexion au moteur interrompue. Vérifiez que le service est démarré puis réessayez.");
  };
  return finish;
}

export async function sendFeedback(searchId: string, interpretationId: string): Promise<boolean> {
  try {
    const response = await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ search_id: searchId, interpretation_id: interpretationId }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export interface BenchValue {
  mean: number;
  low: number;
  high: number;
  n: number;
}

export interface BenchMetric {
  metric: string;
  label: string;
  values: Record<string, BenchValue>;
  difference_vs_rag?: Record<string, { mean: number; low: number; high: number }>;
}

export interface BenchEfficiency {
  queries: number;
  latency_ms: { p50: number; p95: number; mean: number };
  cpu_ms_per_query: number;
  energy_per_query: { value_mwh: number; method: string };
  python_peak_mb: number | null;
  vram_mb: number;
  llm_calls: number;
  tokens_in_per_query: number;
  tokens_out_per_query: number;
  cost_usd_per_query: number | null;
}

export interface BenchDataset {
  dataset: string;
  n: number;
  sampling: string;
  setting: string;
  metrics: BenchMetric[];
  efficiency: Record<string, BenchEfficiency>;
  errors: Record<string, number>;
}

export interface BenchReport {
  version: string;
  generated_at: string;
  host: { platform: string; cpu_count: number; python: string; processor: string };
  llm: { enabled: boolean; status: string };
  datasets: Record<string, BenchDataset>;
  process: { peak_rss_mb: number; vram_mb: number };
  systems: Record<string, string>;
}

export async function loadBenchmarks(): Promise<BenchReport | null> {
  try {
    const response = await fetch("/benchmarks.json", { cache: "no-cache" });
    if (!response.ok) return null;
    return (await response.json()) as BenchReport;
  } catch {
    return null;
  }
}
