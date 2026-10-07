import "./styles.css";

import { DEMO, type Depth, type Possibility, type SearchResult, type Status, demoQueries, sendFeedback, streamSearch } from "./api";
import { renderBenchmarks } from "./landing";
import { NeuralScene, webglAvailable } from "./scene";

const EXAMPLES = [
  "Quel ordinateur à 500 € serait le meilleur pour faire tourner AIOTECH 44 ?",
  "Quel est le délai de rétractation pour un achat en ligne ?",
  "Quelles sont les exceptions au droit de rétractation ?",
  "Quel est le prix du Nova Book 15 ?",
];

const STATUS_HELP: Record<Status, string> = {
  "FAIT": "Cité par au moins une source et sans contradiction ouverte",
  "INFÉRENCE": "Déduit de prémisses vérifiées (t-norme de Gödel)",
  "INCERTAIN": "Sources en désaccord ou contrainte non tranchée",
  "NON VÉRIFIÉ": "Aucune source ne l'établit",
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function must<T extends HTMLElement>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`élément ${selector} absent`);
  return node;
}

const hero = must<HTMLElement>("#app");
const form = must<HTMLFormElement>("#search");
const input = must<HTMLInputElement>("#q");
const depthSelect = must<HTMLSelectElement>("#depth");
const webToggle = must<HTMLInputElement>("#web");
const phase = must<HTMLElement>("#phase");
const answers = must<HTMLElement>("#answers");
const hud = must<HTMLElement>("#hud");
const resetButton = must<HTMLButtonElement>("#reset");
const examples = must<HTMLElement>("#examples");
const canvas = must<HTMLCanvasElement>("#scene");
const labels = must<HTMLElement>("#labels");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let scene: NeuralScene | null = null;
let stop: (() => void) | null = null;
let current: SearchResult | null = null;
let shown: Possibility[] = [];
let searchId = "";
let passageCount = 0;

if (webglAvailable()) {
  try {
    scene = new NeuralScene(canvas, labels, { reducedMotion });
    hero.classList.add("has-webgl");
    scene.onBarMove = (rect) => {
      form.style.left = `${rect.left}px`;
      form.style.top = `${rect.top}px`;
      form.style.width = `${rect.width}px`;
      form.style.height = `${rect.height}px`;
      hero.style.setProperty("--bar-bottom", `${rect.bottom}px`);
      hero.style.setProperty("--bar-h", `${rect.height}px`);
    };
    new IntersectionObserver(([entry]) => scene?.setActive(Boolean(entry?.isIntersecting) && !document.hidden))
      .observe(hero);
  } catch (error) {
    console.warn("WebGL indisponible, interface 2D", error);
    scene = null;
  }
}
if (!scene) {
  hero.classList.add("no-webgl");
  canvas.remove();
  const syncBar = (): void => {
    const rect = form.getBoundingClientRect();
    hero.style.setProperty("--bar-bottom", `${rect.bottom - hero.getBoundingClientRect().top}px`);
    hero.style.setProperty("--bar-h", `${rect.height}px`);
  };
  new ResizeObserver(syncBar).observe(form);
  window.addEventListener("resize", syncBar);
}

function addExamples(queries: string[]): void {
  for (const example of queries) {
    const chip = el("button", "chip", example);
    chip.type = "button";
    chip.addEventListener("click", () => {
      input.value = example;
      run(example);
    });
    examples.append(chip);
  }
}

if (DEMO) {
  must<HTMLElement>("#demo-note").hidden = false;
  must<HTMLElement>(".options").hidden = true;
  void demoQueries().then((queries) => addExamples(queries.map((q) => q.query)));
} else {
  addExamples(EXAMPLES);
}

input.addEventListener("input", () => scene?.typing(Math.min(1, input.value.length / 40)));

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (query) run(query);
});

resetButton.addEventListener("click", reset);
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && hero.dataset.state !== "idle") {
    reset();
    return;
  }
  if (hero.dataset.state === "answers" && document.activeElement !== input && /^[1-3]$/.test(event.key)) {
    const rank = Number(event.key);
    const card = answers.querySelector<HTMLElement>(`[data-rank="${rank}"]`);
    card?.focus();
    card?.scrollIntoView({ block: "nearest", behavior: reducedMotion ? "auto" : "smooth" });
  }
});
window.addEventListener("resize", () => {
  if (hero.dataset.state === "answers") scene?.placeAnswers(anchors());
});

function setPhase(text: string): void {
  phase.textContent = text;
}

const PACE: Record<string, number> = reducedMotion
  ? {}
  : { start: 650, interpretations: 750, retrieval: 220, graph: 450, trajectory: 320, possibilities: 150 };

class Choreography {
  private queue: { kind: string; apply: () => void }[] = [];
  private timer: number | null = null;
  private token = 0;

  push(kind: string, apply: () => void): void {
    this.queue.push({ kind, apply });
    if (this.timer === null) this.next(this.token);
  }

  private next(token: number): void {
    if (token !== this.token) return;
    const step = this.queue.shift();
    if (!step) {
      this.timer = null;
      return;
    }
    step.apply();
    const delay = PACE[step.kind] ?? 0;
    this.timer = window.setTimeout(() => this.next(token), delay);
  }

  cancel(): void {
    this.token += 1;
    this.queue = [];
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = null;
  }
}

const show = new Choreography();

function reset(): void {
  stop?.();
  stop = null;
  show.cancel();
  current = null;
  shown = [];
  hero.dataset.state = "idle";
  answers.replaceChildren();
  hud.replaceChildren();
  setPhase("");
  resetButton.hidden = true;
  scene?.reset();
  input.focus();
}

function run(query: string): void {
  stop?.();
  show.cancel();
  answers.replaceChildren();
  hud.replaceChildren();
  shown = [];
  passageCount = 0;
  hero.dataset.state = "searching";
  resetButton.hidden = false;
  setPhase("Masquage des données personnelles et filtrage des injections…");
  scene?.startQuery();
  stop = streamSearch(query, depthSelect.value as Depth, webToggle.checked, {
    start: (data) => show.push("start", () => {
      searchId = data.search_id;
      const masked = data.pii.reduce((n, p) => n + p.count, 0);
      if (masked > 0) setPhase(`${masked} donnée(s) personnelle(s) masquée(s) avant tout traitement.`);
    }),
    blocked: (data) => show.push("blocked", () => {
      hero.dataset.state = "blocked";
      setPhase(`Requête bloquée : ${data.reason}. Coût : ${data.tokens} jeton, ${data.cost_usd.toFixed(2)} $.`);
      scene?.blocked(data.reason);
    }),
    interpretations: (data) => show.push("interpretations", () => {
      const n = data.neurons.length;
      const constraints = data.constraints.length ? ` · contraintes : ${data.constraints.join(", ")}` : "";
      setPhase(`${n} interprétation${n > 1 ? "s" : ""} de la requête (palier ${data.compute.tier})${constraints}`);
      scene?.showNeurons(data.neurons);
    }),
    retrieval: (data) => show.push("retrieval", () => {
      passageCount += data.passages.length || data.count || 0;
      setPhase(`Recherche hybride : ${passageCount} passages reliés aux interprétations…`);
      scene?.addSources(data.interpretation_id, data.passages, data.count);
    }),
    graph: (data) => show.push("graph", () => {
      const n = data.contradictions.length;
      setPhase(`Graphe : ${data.claims} affirmations, ${data.entities} entités, ${n} contradiction${n > 1 ? "s" : ""} détectée${n > 1 ? "s" : ""}.`);
      scene?.markContradictions(data.contradictions);
    }),
    trajectory: (data) => show.push("trajectory", () => {
      setPhase(`Raisonnement SCG et TAP : trajectoire ${data.interpretation_id} ${data.admissible ? "admissible" : "rejetée"}.`);
      scene?.setTrajectory(data);
    }),
    possibilities: (data) => show.push("possibilities", () => {
      shown = data.possibilities;
    }),
    done: ({ result }) => {
      current = result;
      stop = null;
      show.push("done", () => {
        if (!result.blocked) finish(result);
      });
    },
    failure: (message) => {
      stop = null;
      show.push("failure", () => {
        hero.dataset.state = "error";
        setPhase(message);
      });
    },
  });
}

function finish(result: SearchResult): void {
  const possibilities = result.possibilities.length ? result.possibilities : shown;
  hero.dataset.state = "answers";
  renderHud(result);
  if (!possibilities.length) {
    setPhase("Aucune réponse ne passe la vérification : AIOTECH préfère s'abstenir plutôt que d'affirmer sans preuve.");
    const empty = el("p", "empty", result.warnings.join(" ") || "Essayez une formulation plus précise ou activez le web.");
    answers.append(empty);
    return;
  }
  const n = possibilities.length;
  setPhase(`${n} réponse${n > 1 ? "s" : ""} vérifiée${n > 1 ? "s" : ""}, classée${n > 1 ? "s" : ""} par fiabilité puis par vos choix.`);
  answers.append(...possibilities.map(card));
  if (result.contradictions.length) answers.append(contradictionNote(result));
  if (result.warnings.length) answers.append(el("p", "warnings", result.warnings.join(" ")));
  requestAnimationFrame(() => scene?.contract(possibilities, anchors()));
}

function anchors(): { x: number; y: number }[] {
  const heroRect = hero.getBoundingClientRect();
  const cards = [...answers.querySelectorAll<HTMLElement>(".card")];
  const stacked = cards.length > 1 && cards[1].getBoundingClientRect().top > cards[0].getBoundingClientRect().bottom - 4;
  const top = (answers.getBoundingClientRect().top - heroRect.top) - 46;
  if (stacked) {
    const step = heroRect.width / (cards.length + 1);
    return cards.map((_, i) => ({ x: step * (i + 1), y: top }));
  }
  return cards.map((c) => {
    const r = c.getBoundingClientRect();
    return { x: r.left - heroRect.left + r.width / 2, y: top };
  });
}

function card(p: Possibility): HTMLElement {
  const article = el("article", "card");
  article.tabIndex = 0;
  article.dataset.rank = String(p.rank);
  article.style.setProperty("--rel", String(p.reliability_pct));
  article.dataset.level = p.reliability >= 0.75 ? "high" : p.reliability >= 0.5 ? "mid" : "low";

  const header = el("header");
  header.append(el("span", "rank", `#${p.rank}`), el("span", "interp", p.interpretation_label));
  const rel = el("span", "rel", `${p.reliability_pct} %`);
  rel.title = "Fiabilité : min(admissibilité SCG, qualité TAP)";
  header.append(rel);
  const meter = el("div", "meter");
  meter.setAttribute("role", "meter");
  meter.setAttribute("aria-valuemin", "0");
  meter.setAttribute("aria-valuemax", "100");
  meter.setAttribute("aria-valuenow", String(p.reliability_pct));
  meter.setAttribute("aria-label", "Fiabilité");
  meter.append(el("span"));

  const claims = el("details", "claims");
  claims.append(el("summary", undefined, `Affirmations vérifiées (${p.claims.length})`));
  const list = el("ul");
  for (const c of p.claims) {
    const item = el("li");
    const badge = el("span", `badge s-${c.status.normalize("NFD").replace(/[^A-Za-z]/g, "").toLowerCase()}`, c.status);
    badge.title = STATUS_HELP[c.status];
    const reason = el("small", undefined, c.reason);
    item.append(badge, el("span", "claim-text", c.text), reason);
    list.append(item);
  }
  claims.append(list);

  const sources = el("p", "sources");
  sources.append(el("span", undefined, "Sources : "));
  p.sources.slice(0, 4).forEach((s, i) => {
    if (i > 0) sources.append(", ");
    if (s.url && /^https?:\/\//.test(s.url)) {
      const a = el("a", undefined, s.title);
      a.href = s.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer nofollow";
      sources.append(a);
    } else {
      sources.append(el("span", undefined, s.title));
    }
  });

  const choose = el("button", "choose", "C'est ce que je cherchais");
  choose.type = "button";
  choose.addEventListener("click", async () => {
    choose.disabled = true;
    scene?.highlight(p.rank);
    for (const other of answers.querySelectorAll(".card")) other.classList.toggle("chosen", other === article);
    const ok = searchId ? await sendFeedback(searchId, p.interpretation_id) : false;
    choose.textContent = ok ? "Merci : ce choix départagera les réponses à fiabilité égale"
      : DEMO ? "Vitrine : sur votre serveur, ce choix départage les réponses à fiabilité égale" : "Choix non enregistré";
  });

  article.append(header, meter, el("h3", undefined, p.title));
  if (p.answer.trim() !== p.title.trim()) article.append(el("p", "answer", p.answer));
  article.append(claims, sources);
  if (p.alternatives.length) article.append(el("p", "alternatives", `Autres candidats : ${p.alternatives.join(", ")}`));
  article.append(choose);
  return article;
}

function contradictionNote(result: SearchResult): HTMLElement {
  const box = el("aside", "contradictions");
  box.append(el("h4", undefined, "Contradictions détectées entre sources"));
  const list = el("ul");
  for (const c of result.contradictions) {
    const values = c.groups.map((g) => g.value_text).join(" ≠ ");
    const verdict = c.resolved && c.winner ? `tranchée pour ${c.winner} (sources plus fiables)` : "non tranchée : marquée [INCERTAIN]";
    list.append(el("li", undefined, `${c.entity_label} · ${c.attribute} : ${values}, ${verdict}`));
  }
  box.append(list);
  return box;
}

function renderHud(result: SearchResult): void {
  const latency = Number(result.metrics.latency_ms ?? 0);
  const cost = result.usage.cost_usd;
  const parts = [
    `${latency.toFixed(0)} ms`,
    `${result.usage.llm_calls} appel${result.usage.llm_calls > 1 ? "s" : ""} LLM`,
    `${result.usage.tokens_in + result.usage.tokens_out} jetons`,
    cost === null ? "coût inconnu" : `${cost.toFixed(4)} $`,
  ];
  if (result.metrics.cached) parts.push("depuis le cache");
  hud.replaceChildren(...parts.map((part) => el("span", undefined, part)));
}

hero.dataset.state = "idle";
void renderBenchmarks(must<HTMLElement>("#bench"));
if (current === null && window.location.hash === "") input.focus({ preventScroll: true });
