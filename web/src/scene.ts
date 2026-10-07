import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

import type { Contradiction, Neuron, PassageRef, Possibility, TrajectoryEvent } from "./api";

const COLOR = {
  cyan: new THREE.Color(0x5fe1ff),
  violet: new THREE.Color(0x9b8cff),
  emerald: new THREE.Color(0x34d399),
  amber: new THREE.Color(0xfbbf24),
  orange: new THREE.Color(0xfb923c),
  red: new THREE.Color(0xf87171),
  core: new THREE.Color(0xe8f7ff),
  dim: new THREE.Color(0x3b4252),
};

const BAR_WIDTH = 7.6;
const BAR_HEIGHT = 1.05;
const BAR_DEPTH = 0.22;
const SOURCE_LIMIT = 8;

type Ease = (t: number) => number;
const easeInOut: Ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const easeOut: Ease = (t) => 1 - (1 - t) ** 3;

export function reliabilityColor(reliability: number): THREE.Color {
  if (reliability >= 0.75) return COLOR.emerald.clone();
  if (reliability >= 0.5) return COLOR.amber.clone();
  return COLOR.orange.clone();
}

interface Tween {
  elapsed: number;
  duration: number;
  ease: Ease;
  update: (k: number) => void;
  done?: () => void;
}

class Tweens {
  private items: Tween[] = [];

  constructor(private readonly instant: () => boolean) {}

  add(duration: number, update: (k: number) => void, ease: Ease = easeInOut, done?: () => void): void {
    if (this.instant() || duration <= 0) {
      update(1);
      done?.();
      return;
    }
    this.items.push({ elapsed: 0, duration, ease, update, done });
  }

  step(dt: number): void {
    const running = this.items;
    this.items = [];
    for (const tween of running) {
      tween.elapsed += dt;
      const t = Math.min(1, tween.elapsed / tween.duration);
      tween.update(tween.ease(t));
      if (t < 1) this.items.push(tween);
      else tween.done?.();
    }
  }

  clear(): void {
    this.items = [];
  }
}

function glowTexture(): THREE.Texture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.22, "rgba(255,255,255,0.55)");
    g.addColorStop(0.55, "rgba(255,255,255,0.12)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function sprite(texture: THREE.Texture, color: THREE.Color, scale: number, opacity = 1): THREE.Sprite {
  const material = new THREE.SpriteMaterial({
    map: texture,
    color,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const s = new THREE.Sprite(material);
  s.scale.setScalar(scale);
  return s;
}

class Node3D {
  readonly group = new THREE.Group();
  readonly core: THREE.Mesh<THREE.IcosahedronGeometry, THREE.MeshBasicMaterial>;
  readonly halo: THREE.Sprite;
  readonly shell: THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial>;
  readonly labelEl: HTMLDivElement;
  readonly label: CSS2DObject;
  color: THREE.Color;
  opacity = 1;
  spin = Math.random() * 0.6 + 0.2;

  constructor(texture: THREE.Texture, color: THREE.Color, radius: number, title: string, meta: string) {
    this.color = color.clone();
    this.core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(radius, 3),
      new THREE.MeshBasicMaterial({ color: this.color, transparent: true }),
    );
    this.halo = sprite(texture, this.color, radius * 7, 0.85);
    this.shell = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(radius * 1.9, 1)),
      new THREE.LineBasicMaterial({ color: this.color, transparent: true, opacity: 0.35, depthWrite: false,
        blending: THREE.AdditiveBlending }),
    );
    this.group.add(this.halo, this.core, this.shell);
    this.labelEl = document.createElement("div");
    this.labelEl.className = "neuron-label";
    this.setLabel(title, meta);
    this.label = new CSS2DObject(this.labelEl);
    this.label.position.set(0, -radius * 2.6, 0);
    this.group.add(this.label);
  }

  setLabel(title: string, meta: string, tone = ""): void {
    this.labelEl.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = title;
    const span = document.createElement("span");
    span.textContent = meta;
    this.labelEl.append(strong, span);
    this.labelEl.dataset.tone = tone;
  }

  setColor(color: THREE.Color): void {
    this.color.copy(color);
    this.core.material.color.copy(color);
    this.halo.material.color.copy(color);
    this.shell.material.color.copy(color);
  }

  setOpacity(opacity: number): void {
    this.opacity = opacity;
    this.core.material.opacity = opacity;
    this.halo.material.opacity = 0.85 * opacity;
    this.shell.material.opacity = 0.35 * opacity;
    this.labelEl.style.opacity = String(Math.min(1, opacity * 1.2));
  }

  tick(dt: number, time: number): void {
    this.shell.rotation.y += dt * this.spin;
    this.shell.rotation.x += dt * this.spin * 0.4;
    const breathe = 1 + Math.sin(time * 2.2 + this.spin * 10) * 0.06;
    this.halo.scale.setScalar(this.core.geometry.parameters.radius * 7 * breathe);
  }

  dispose(): void {
    this.labelEl.remove();
    this.core.geometry.dispose();
    this.core.material.dispose();
    this.halo.material.dispose();
    this.shell.geometry.dispose();
    this.shell.material.dispose();
  }
}

class Synapse {
  readonly line: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  private readonly curve = new THREE.QuadraticBezierCurve3();
  private readonly positions: Float32Array;
  private readonly bend: THREE.Vector3;
  private readonly pulses: { sprite: THREE.Sprite; t: number; speed: number }[] = [];
  readonly from: () => THREE.Vector3;
  readonly to: () => THREE.Vector3;
  baseOpacity: number;

  constructor(
    from: () => THREE.Vector3,
    to: () => THREE.Vector3,
    color: THREE.Color,
    opacity: number,
    texture: THREE.Texture | null,
    pulseCount: number,
    pulseScale = 0.28,
  ) {
    this.from = from;
    this.to = to;
    this.baseOpacity = opacity;
    this.positions = new Float32Array(33 * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false,
        blending: THREE.AdditiveBlending }),
    );
    this.bend = new THREE.Vector3((Math.random() - 0.5) * 1.6, (Math.random() - 0.5) * 1.6, (Math.random() - 0.5) * 1.6);
    if (texture) {
      for (let i = 0; i < pulseCount; i += 1) {
        const s = sprite(texture, color, pulseScale, 0.9);
        this.line.add(s);
        this.pulses.push({ sprite: s, t: Math.random(), speed: 0.35 + Math.random() * 0.35 });
      }
    }
    this.update(0);
  }

  setColor(color: THREE.Color): void {
    this.line.material.color.copy(color);
    for (const p of this.pulses) p.sprite.material.color.copy(color);
  }

  setOpacity(opacity: number): void {
    this.line.material.opacity = opacity;
    for (const p of this.pulses) p.sprite.material.opacity = Math.min(1, opacity * 2.2);
  }

  update(dt: number): void {
    const a = this.from();
    const b = this.to();
    const mid = a.clone().add(b).multiplyScalar(0.5).add(this.bend);
    this.curve.v0.copy(a);
    this.curve.v1.copy(mid);
    this.curve.v2.copy(b);
    for (let i = 0; i <= 32; i += 1) {
      const p = this.curve.getPoint(i / 32);
      this.positions[i * 3] = p.x;
      this.positions[i * 3 + 1] = p.y;
      this.positions[i * 3 + 2] = p.z;
    }
    this.line.geometry.attributes.position.needsUpdate = true;
    this.line.geometry.computeBoundingSphere();
    for (const pulse of this.pulses) {
      pulse.t = (pulse.t + dt * pulse.speed) % 1;
      pulse.sprite.position.copy(this.curve.getPoint(pulse.t));
    }
  }

  dispose(): void {
    this.line.geometry.dispose();
    this.line.material.dispose();
    for (const p of this.pulses) p.sprite.material.dispose();
  }
}

interface SourceDot {
  id: string;
  documentId: string;
  mesh: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  synapse: Synapse;
  offset: THREE.Vector3;
}

interface NeuronEntry {
  data: Neuron;
  node: Node3D;
  home: THREE.Vector3;
  synapse: Synapse;
  sources: SourceDot[];
  state: "pending" | "admissible" | "rejected";
}

interface Spark {
  sprite: THREE.Sprite;
  velocity: THREE.Vector3;
  life: number;
}

export interface SceneOptions {
  reducedMotion: boolean;
}

export class NeuralScene {
  readonly renderer: THREE.WebGLRenderer;
  private readonly labels: CSS2DRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  private readonly clock = new THREE.Clock();
  private readonly tweens: Tweens;
  private readonly texture = glowTexture();
  private readonly field = new THREE.Group();
  private readonly cloud = new THREE.Group();
  private readonly bar = new THREE.Group();
  private readonly barEdge: THREE.LineLoop<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  private readonly barSweep: THREE.Sprite;
  private readonly coreNode: Node3D;
  private readonly shock: THREE.Mesh<THREE.TorusGeometry, THREE.MeshBasicMaterial>;
  private neurons = new Map<string, NeuronEntry>();
  private answers: { node: Node3D; synapse: Synapse; rank: number; interpretationId: string }[] = [];
  private conflicts: Synapse[] = [];
  private sparks: Spark[] = [];
  private pointer = new THREE.Vector2();
  private running = true;
  private time = 0;
  private barTarget = { y: 0, scale: 1 };
  private readonly reduced: boolean;
  onBarMove: ((rect: DOMRect) => void) | null = null;

  constructor(canvas: HTMLCanvasElement, labelLayer: HTMLElement, options: SceneOptions) {
    this.reduced = options.reducedMotion;
    this.tweens = new Tweens(() => this.reduced);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.labels = new CSS2DRenderer({ element: labelLayer });
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    this.camera.position.set(0, 0, 16);

    this.buildField();
    this.scene.add(this.field, this.cloud, this.bar);

    const shape = new THREE.Shape();
    const w = BAR_WIDTH / 2;
    const h = BAR_HEIGHT / 2;
    const r = h;
    shape.moveTo(-w + r, -h);
    shape.lineTo(w - r, -h);
    shape.absarc(w - r, 0, r, -Math.PI / 2, Math.PI / 2, false);
    shape.lineTo(-w + r, h);
    shape.absarc(-w + r, 0, r, Math.PI / 2, (3 * Math.PI) / 2, false);
    const slab = new THREE.Mesh(
      new THREE.ExtrudeGeometry(shape, {
        depth: BAR_DEPTH, bevelEnabled: true, bevelThickness: 0.06, bevelSize: 0.06, bevelSegments: 5, curveSegments: 40,
      }).translate(0, 0, -BAR_DEPTH / 2),
      new THREE.MeshPhysicalMaterial({
        color: 0x0b1828, metalness: 0.15, roughness: 0.14, clearcoat: 1, clearcoatRoughness: 0.08,
        transparent: true, opacity: 0.86, envMapIntensity: 0.7, sheen: 0.6, sheenColor: new THREE.Color(0x5fe1ff),
      }),
    );
    this.barEdge = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(
        shape.getPoints(80).map((p) => new THREE.Vector3(p.x * 1.012, p.y * 1.05, BAR_DEPTH / 2 + 0.07)),
      ),
      new THREE.LineBasicMaterial({ color: COLOR.cyan, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending }),
    );
    this.barSweep = sprite(this.texture, COLOR.cyan, 2.4, 0.35);
    this.barSweep.scale.set(2.2, 1.3, 1);
    this.barSweep.position.set(-w, 0, BAR_DEPTH / 2 + 0.1);
    const underGlow = sprite(this.texture, COLOR.violet, 1, 0.35);
    underGlow.scale.set(BAR_WIDTH * 1.4, 2.6, 1);
    underGlow.position.set(0, 0, -0.4);
    this.bar.add(underGlow, slab, this.barEdge, this.barSweep);
    const key = new THREE.PointLight(0x9fe8ff, 30, 30);
    key.position.set(-4, 4, 6);
    this.scene.add(key, new THREE.AmbientLight(0x6070a0, 0.6));

    this.coreNode = new Node3D(this.texture, COLOR.core, 0.32, "", "");
    this.coreNode.labelEl.classList.add("core-label");
    this.coreNode.label.visible = false;
    this.coreNode.group.visible = false;
    this.scene.add(this.coreNode.group);
    this.shock = new THREE.Mesh(
      new THREE.TorusGeometry(1, 0.03, 8, 96),
      new THREE.MeshBasicMaterial({ color: COLOR.red, transparent: true, opacity: 0, blending: THREE.AdditiveBlending }),
    );
    this.scene.add(this.shock);

    window.addEventListener("pointermove", (event) => {
      this.pointer.set((event.clientX / window.innerWidth) * 2 - 1, (event.clientY / window.innerHeight) * 2 - 1);
    });
    document.addEventListener("visibilitychange", () => this.setActive(!document.hidden));
    this.resize();
    window.addEventListener("resize", () => this.resize());
    this.loop();
  }

  private buildField(): void {
    const count = this.reduced ? 500 : 1400;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const points: THREE.Vector3[] = [];
    for (let i = 0; i < count; i += 1) {
      const radius = 9 + Math.random() * 26;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const p = new THREE.Vector3(
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.sin(phi) * Math.sin(theta) * 0.7,
        radius * Math.cos(phi) - 10,
      );
      points.push(p);
      positions.set([p.x, p.y, p.z], i * 3);
      const c = COLOR.cyan.clone().lerp(COLOR.violet, Math.random());
      colors.set([c.r, c.g, c.b], i * 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.14, map: this.texture, vertexColors: true, transparent: true, opacity: 0.75, depthWrite: false,
      blending: THREE.AdditiveBlending, sizeAttenuation: true,
    });
    this.field.add(new THREE.Points(geometry, material));
    const segments: number[] = [];
    for (let i = 0; i < points.length && segments.length < 6 * 160; i += 1) {
      for (let j = i + 1; j < Math.min(points.length, i + 40); j += 1) {
        if (points[i].distanceTo(points[j]) < 3.2) {
          segments.push(points[i].x, points[i].y, points[i].z, points[j].x, points[j].y, points[j].z);
          break;
        }
      }
    }
    const web = new THREE.BufferGeometry();
    web.setAttribute("position", new THREE.Float32BufferAttribute(segments, 3));
    this.field.add(new THREE.LineSegments(web, new THREE.LineBasicMaterial({
      color: COLOR.cyan, transparent: true, opacity: 0.07, depthWrite: false, blending: THREE.AdditiveBlending,
    })));
  }

  setActive(active: boolean): void {
    if (active === this.running) return;
    this.running = active;
    if (active) {
      this.clock.getDelta();
      this.loop();
    }
  }

  private get width(): number {
    return this.renderer.domElement.clientWidth || window.innerWidth;
  }

  private get height(): number {
    return this.renderer.domElement.clientHeight || window.innerHeight;
  }

  private halfExtents(): { hw: number; hh: number } {
    const hh = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * this.camera.position.z;
    return { hw: hh * this.camera.aspect, hh };
  }

  resize(): void {
    const width = this.width;
    const height = this.height;
    this.camera.aspect = width / height;
    const needed = (BAR_WIDTH * 1.12) / 2 / this.camera.aspect / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    this.camera.position.z = Math.max(14, needed);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.labels.setSize(width, height);
    this.applyBarTarget();
  }

  private applyBarTarget(): void {
    const { hh } = this.halfExtents();
    this.bar.position.y = this.barTarget.y * hh;
    this.bar.scale.setScalar(this.barTarget.scale);
  }

  private moveBar(y: number, scale: number, duration: number): void {
    const from = { ...this.barTarget };
    this.tweens.add(duration, (k) => {
      this.barTarget = { y: from.y + (y - from.y) * k, scale: from.scale + (scale - from.scale) * k };
      this.applyBarTarget();
    });
  }

  barScreenRect(): DOMRect {
    this.bar.updateMatrixWorld();
    const w = BAR_WIDTH / 2;
    const h = BAR_HEIGHT / 2;
    const corners = [new THREE.Vector3(-w, h, BAR_DEPTH / 2), new THREE.Vector3(w, -h, BAR_DEPTH / 2)]
      .map((v) => this.bar.localToWorld(v).project(this.camera));
    const x0 = ((corners[0].x + 1) / 2) * this.width;
    const y0 = ((1 - corners[0].y) / 2) * this.height;
    const x1 = ((corners[1].x + 1) / 2) * this.width;
    const y1 = ((1 - corners[1].y) / 2) * this.height;
    return new DOMRect(x0, y0, x1 - x0, y1 - y0);
  }

  worldAt(screenX: number, screenY: number): THREE.Vector3 {
    const ndc = new THREE.Vector3((screenX / this.width) * 2 - 1, -(screenY / this.height) * 2 + 1, 0.5);
    ndc.unproject(this.camera);
    const dir = ndc.sub(this.camera.position).normalize();
    const distance = -this.camera.position.z / dir.z;
    return this.camera.position.clone().add(dir.multiplyScalar(distance));
  }

  typing(intensity: number): void {
    if (this.reduced) return;
    const count = Math.min(6, 1 + Math.round(intensity * 4));
    for (let i = 0; i < count; i += 1) {
      const color = Math.random() < 0.5 ? COLOR.cyan : COLOR.violet;
      const s = sprite(this.texture, color, 0.22 + Math.random() * 0.2, 0.95);
      const x = (Math.random() - 0.5) * BAR_WIDTH * this.bar.scale.x;
      s.position.set(x, this.bar.position.y + (Math.random() - 0.5) * 0.6, 0.3);
      const velocity = new THREE.Vector3((Math.random() - 0.5) * 2.5, (Math.random() - 0.2) * 2.5, -3 - Math.random() * 4);
      this.scene.add(s);
      this.sparks.push({ sprite: s, velocity, life: 1.3 });
    }
  }

  private clearQuery(): void {
    for (const entry of this.neurons.values()) {
      this.cloud.remove(entry.node.group, entry.synapse.line);
      entry.node.dispose();
      entry.synapse.dispose();
      for (const dot of entry.sources) {
        this.cloud.remove(dot.mesh, dot.synapse.line);
        dot.mesh.geometry.dispose();
        dot.mesh.material.dispose();
        dot.synapse.dispose();
      }
    }
    this.neurons.clear();
    for (const answer of this.answers) {
      this.cloud.remove(answer.node.group, answer.synapse.line);
      answer.node.dispose();
      answer.synapse.dispose();
    }
    this.answers = [];
    for (const conflict of this.conflicts) {
      this.cloud.remove(conflict.line);
      conflict.dispose();
    }
    this.conflicts = [];
  }

  startQuery(): void {
    this.tweens.clear();
    this.clearQuery();
    this.moveBar(0.78, 0.62, 0.9);
    this.coreNode.group.visible = true;
    this.coreNode.setColor(COLOR.core);
    this.coreNode.label.visible = false;
    this.coreNode.group.position.set(0, 0.2, 0);
    this.coreNode.group.scale.setScalar(0.01);
    this.tweens.add(0.8, (k) => this.coreNode.group.scale.setScalar(0.01 + k), easeOut);
    this.coreNode.setOpacity(1);
    this.cloud.visible = true;
  }

  showNeurons(neurons: Neuron[]): void {
    const { hw, hh } = this.halfExtents();
    const radius = Math.min(hw * 0.62, hh * 0.55, 6);
    const center = this.coreNode.group.position;
    const golden = Math.PI * (3 - Math.sqrt(5));
    neurons.forEach((data, index) => {
      const y = 1 - ((index + 0.5) / neurons.length) * 2;
      const ring = Math.sqrt(1 - y * y);
      const angle = index * golden + 0.6;
      const portrait = this.camera.aspect < 1;
      const home = new THREE.Vector3(Math.cos(angle) * ring * radius * (portrait ? 0.9 : 1.25),
        y * radius * (portrait ? 1.6 : 0.55), Math.sin(angle) * ring * radius * 0.6).add(center);
      const size = 0.18 + data.plausibility * 0.2;
      const node = new Node3D(this.texture, COLOR.cyan, size, data.label, `plausibilité ${Math.round(data.plausibility * 100)} %`);
      node.group.position.copy(center);
      node.setOpacity(0);
      const synapse = new Synapse(() => this.coreNode.group.position, () => node.group.position, COLOR.cyan, 0.5,
        this.reduced ? null : this.texture, 3);
      synapse.setOpacity(0);
      this.cloud.add(synapse.line, node.group);
      this.neurons.set(data.id, { data, node, home, synapse, sources: [], state: "pending" });
      this.tweens.add(1.0 + index * 0.12, (k) => {
        node.group.position.lerpVectors(center, home, k);
        node.setOpacity(k);
        synapse.setOpacity(0.5 * k);
      }, easeOut);
    });
  }

  addSources(interpretationId: string, passages: PassageRef[], count?: number): void {
    const entry = this.neurons.get(interpretationId);
    if (!entry) return;
    const refs = passages.length > 0 ? passages.slice(0, SOURCE_LIMIT)
      : Array.from({ length: Math.min(count ?? 0, SOURCE_LIMIT) }, (_, i) => ({
        id: `${interpretationId}-${i}`, title: "", url: null, origin: "corpus" as const, reliability: 0.6, score: 0,
      }));
    refs.forEach((ref, index) => {
      const color = ref.origin === "web" ? COLOR.violet : COLOR.cyan;
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.05 + ref.reliability * 0.06, 12, 12),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0 }),
      );
      const dir = entry.home.clone().sub(this.coreNode.group.position).normalize();
      const offset = new THREE.Vector3().randomDirection().multiplyScalar(0.9 + Math.random() * 0.8).add(dir.multiplyScalar(0.6));
      mesh.position.copy(entry.node.group.position);
      const synapse = new Synapse(() => mesh.position, () => entry.node.group.position, color, 0.25,
        this.reduced ? null : this.texture, 1, 0.16);
      synapse.setOpacity(0);
      this.cloud.add(synapse.line, mesh);
      const dot: SourceDot = { id: ref.id, documentId: ref.id.split("#")[0], mesh, synapse, offset };
      entry.sources.push(dot);
      this.tweens.add(0.7 + index * 0.06, (k) => {
        mesh.position.copy(entry.node.group.position).addScaledVector(offset, k);
        mesh.material.opacity = (0.35 + ref.reliability * 0.65) * k;
        synapse.setOpacity(0.25 * k);
      }, easeOut);
    });
  }

  markContradictions(contradictions: Contradiction[]): void {
    const dots = [...this.neurons.values()].flatMap((entry) => entry.sources);
    for (const contradiction of contradictions) {
      const groups = contradiction.groups.map((g) => dots.filter((d) => g.source_ids.includes(d.documentId)));
      for (let i = 0; i < groups.length; i += 1) {
        for (let j = i + 1; j < groups.length; j += 1) {
          const a = groups[i][0];
          const b = groups[j][0];
          if (!a || !b) continue;
          const color = contradiction.resolved ? COLOR.amber : COLOR.red;
          const arc = new Synapse(() => a.mesh.position, () => b.mesh.position, color, 0.8,
            this.reduced ? null : this.texture, 2, 0.2);
          this.conflicts.push(arc);
          this.cloud.add(arc.line);
        }
      }
    }
  }

  setTrajectory(event: TrajectoryEvent): void {
    const entry = this.neurons.get(event.interpretation_id);
    if (!entry) return;
    const { node, synapse } = entry;
    if (event.admissible && event.best) {
      entry.state = "admissible";
      const color = reliabilityColor(event.best.reliability);
      const start = node.color.clone();
      this.tweens.add(0.6, (k) => {
        const c = start.clone().lerp(color, k);
        node.setColor(c);
        synapse.setColor(c);
        node.group.scale.setScalar(1 + 0.18 * k);
      });
      node.setLabel(entry.data.label, `${event.best.label} · fiabilité ${Math.round(event.best.reliability * 100)} %`, "ok");
    } else {
      entry.state = "rejected";
      const start = node.color.clone();
      this.tweens.add(0.7, (k) => {
        node.setColor(start.clone().lerp(COLOR.red, k));
        synapse.setColor(start.clone().lerp(COLOR.red, k));
        node.group.scale.setScalar(1 - 0.4 * k);
        node.setOpacity(1 - 0.55 * k);
        synapse.setOpacity(0.5 - 0.42 * k);
      });
      const reason = (event.rejection_reason ?? "aucune réponse vérifiée").replace(/^Prémisse fausse : /, "");
      node.setLabel(entry.data.label, `rejetée : ${reason.length > 70 ? `${reason.slice(0, 67)}…` : reason}`, "rejected");
    }
  }

  contract(possibilities: Possibility[], anchors: { x: number; y: number }[]): void {
    const used = new Set<string>();
    this.answers = possibilities.map((possibility, index) => {
      const origin = this.neurons.get(possibility.interpretation_id);
      const start = origin ? origin.node.group.position.clone() : this.coreNode.group.position.clone();
      const color = reliabilityColor(possibility.reliability);
      const node = new Node3D(this.texture, color, index === 0 ? 0.2 : 0.16, `#${possibility.rank}`, "");
      node.label.visible = false;
      node.group.position.copy(start);
      const synapse = new Synapse(() => this.coreNode.group.position, () => node.group.position, color, 0.6,
        this.reduced ? null : this.texture, 4, 0.3);
      this.cloud.add(synapse.line, node.group);
      used.add(possibility.interpretation_id);
      return { node, synapse, rank: possibility.rank, interpretationId: possibility.interpretation_id };
    });
    for (const entry of this.neurons.values()) {
      const fading: THREE.Object3D[] = [entry.node.group, ...entry.sources.map((d) => d.mesh)];
      const center = this.coreNode.group.position.clone();
      const from = entry.node.group.position.clone();
      this.tweens.add(1.1, (k) => {
        entry.node.group.position.lerpVectors(from, center, k * 0.7);
        entry.node.setOpacity((1 - k) * (used.has(entry.data.id) ? 0.6 : 0.5));
        entry.synapse.setOpacity(0.4 * (1 - k));
        for (const dot of entry.sources) {
          dot.mesh.material.opacity *= 1 - k * 0.5;
          dot.synapse.setOpacity(0.25 * (1 - k));
        }
      }, easeInOut, () => {
        entry.node.group.visible = false;
        for (const item of fading) item.visible = false;
        for (const dot of entry.sources) dot.synapse.line.visible = false;
        entry.synapse.line.visible = false;
      });
    }
    for (const conflict of this.conflicts) {
      this.tweens.add(1.0, (k) => conflict.setOpacity(0.8 * (1 - k)), easeInOut, () => { conflict.line.visible = false; });
    }
    this.placeAnswers(anchors, 1.3);
  }

  placeAnswers(anchors: { x: number; y: number }[], duration = 0): void {
    const bar = this.barScreenRect();
    const coreTarget = this.worldAt(bar.left + bar.width / 2, bar.top + bar.height / 2);
    const coreFrom = this.coreNode.group.position.clone();
    this.tweens.add(duration, (k) => {
      this.coreNode.group.position.lerpVectors(coreFrom, coreTarget, k);
      this.coreNode.group.scale.setScalar(1 - 0.7 * k);
    });
    this.answers.forEach((answer, index) => {
      const anchor = anchors[index];
      if (!anchor) return;
      const target = this.worldAt(anchor.x, anchor.y);
      const from = answer.node.group.position.clone();
      this.tweens.add(duration + index * 0.1, (k) => answer.node.group.position.lerpVectors(from, target, k), easeOut);
    });
  }

  highlight(rank: number): void {
    for (const answer of this.answers) {
      const chosen = answer.rank === rank;
      answer.node.group.scale.setScalar(chosen ? 1.25 : 0.9);
      answer.synapse.setOpacity(chosen ? 0.95 : 0.35);
    }
    if (this.reduced) return;
    const target = this.answers.find((a) => a.rank === rank);
    if (!target) return;
    for (let i = 0; i < 18; i += 1) {
      const s = sprite(this.texture, target.node.color, 0.3, 1);
      s.position.copy(target.node.group.position);
      this.scene.add(s);
      this.sparks.push({ sprite: s, velocity: new THREE.Vector3().randomDirection().multiplyScalar(2.5), life: 0.9 });
    }
  }

  blocked(reason: string): void {
    this.coreNode.group.visible = true;
    this.coreNode.setColor(COLOR.red);
    this.coreNode.setLabel("Requête bloquée", reason, "rejected");
    this.coreNode.label.visible = true;
    this.coreNode.group.scale.setScalar(1);
    this.shock.position.copy(this.coreNode.group.position);
    this.tweens.add(1.2, (k) => {
      this.shock.scale.setScalar(0.3 + k * 6);
      this.shock.material.opacity = 0.9 * (1 - k);
    }, easeOut);
    const baseX = this.bar.position.x;
    this.tweens.add(0.5, (k) => {
      this.bar.position.x = baseX + Math.sin(k * Math.PI * 6) * 0.25 * (1 - k);
    }, (t) => t);
  }

  reset(): void {
    this.tweens.clear();
    this.clearQuery();
    this.coreNode.group.visible = false;
    this.shock.material.opacity = 0;
    this.bar.position.x = 0;
    this.moveBar(0, 1, 0.8);
  }

  private loop = (): void => {
    if (!this.running) return;
    requestAnimationFrame(this.loop);
    const dt = Math.min(0.05, this.clock.getDelta());
    this.time += dt;
    this.tweens.step(dt);
    if (!this.reduced) {
      this.field.rotation.y += dt * 0.012;
      this.field.rotation.x = THREE.MathUtils.lerp(this.field.rotation.x, this.pointer.y * 0.05, 0.03);
      this.camera.position.x = THREE.MathUtils.lerp(this.camera.position.x, this.pointer.x * 0.6, 0.04);
      this.camera.position.y = THREE.MathUtils.lerp(this.camera.position.y, -this.pointer.y * 0.35, 0.04);
      this.camera.lookAt(0, 0, 0);
      const sweep = (this.time * 0.22) % 1.6;
      this.barSweep.position.x = (sweep - 0.3) * BAR_WIDTH * 0.9 - BAR_WIDTH / 2;
      this.barSweep.material.opacity = sweep < 1.2 ? 0.28 : 0;
      this.barEdge.material.opacity = 0.75 + Math.sin(this.time * 2) * 0.15;
    }
    this.bar.rotation.x = Math.atan2(this.bar.position.y - this.camera.position.y, this.camera.position.z);
    this.bar.rotation.y = Math.atan2(this.camera.position.x - this.bar.position.x, this.camera.position.z) * 0.5;
    this.coreNode.tick(dt, this.time);
    for (const entry of this.neurons.values()) {
      entry.node.tick(dt, this.time);
      entry.synapse.update(dt);
      for (const dot of entry.sources) dot.synapse.update(dt);
    }
    for (const answer of this.answers) {
      answer.node.tick(dt, this.time);
      answer.synapse.update(dt);
    }
    for (const conflict of this.conflicts) {
      conflict.update(dt);
      if (!this.reduced && conflict.line.visible) conflict.line.material.opacity = 0.45 + Math.abs(Math.sin(this.time * 7)) * 0.45;
    }
    this.sparks = this.sparks.filter((spark) => {
      spark.life -= dt;
      spark.sprite.position.addScaledVector(spark.velocity, dt);
      spark.sprite.material.opacity = Math.max(0, spark.life);
      if (spark.life > 0) return true;
      this.scene.remove(spark.sprite);
      spark.sprite.material.dispose();
      return false;
    });
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
    this.onBarMove?.(this.barScreenRect());
  };
}

export function webglAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(window.WebGL2RenderingContext && canvas.getContext("webgl2"));
  } catch {
    return false;
  }
}
