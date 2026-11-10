"use strict";

// A deterministic illustration, not competition data or an engine replay.
const canvas = document.querySelector("#field-canvas");
const durationInput = document.querySelector("#duration");
const weatherButtons = [...document.querySelectorAll("[data-weather]")];
let weather = 0.7;
let randomState = 2606;
function random() {
  randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
  return randomState / 4294967296;
}
const targets = [];
for (let row = 0; row < 4; row++) {
  for (let col = 0; col < 4; col++) {
    for (let j = 0; j < 4; j++) {
      targets.push({
        row,
        col,
        x: (col + 0.14 + random() * 0.72) / 4,
        y: (row + 0.14 + random() * 0.72) / 4,
        flux: 0.15 + random() * 0.85,
        weight: 0.7 + random() * 1.1,
        model: 0.82 + random() * 0.36,
        best: j === 3 ? 1.45 : random() * 0.64,
      });
    }
  }
}
const skyStars = Array.from({ length: 90 }, () => ({
  x: random(),
  y: random(),
  r: 0.3 + random() * 0.8,
  opacity: 0.12 + random() * 0.3,
}));
const programs = [
  { name: "DARK", multiplier: 1.45, band: 0 },
  { name: "BRIGHT", multiplier: 1.15, band: 1 },
  { name: "BACKUP", multiplier: 1, band: 2 },
];
function bandOf(sky) {
  return sky >= 0.65 ? 0 : sky >= 0.4 ? 1 : 2;
}
function evaluate(seconds) {
  // Jointly compare program and one-target-per-cell assignments for a fixed field.
  // The real planner also searches pointings and prices the remaining season.
  let bestPlan = null;
  for (const program of programs) {
    const cells = new Map();
    targets.forEach((target, index) => {
      const reach = Math.min(
        1,
        (target.flux * weather * target.model * seconds) / 450,
      );
      const multiplier =
        bandOf(weather * target.model) === program.band
          ? program.multiplier
          : 1;
      const gain =
        target.weight * Math.max(0, reach * multiplier - target.best);
      const cell = target.row * 4 + target.col;
      if (gain > 0.004 && (!cells.has(cell) || gain > cells.get(cell).gain))
        cells.set(cell, { index, gain, reach });
    });
    const gain = [...cells.values()].reduce(
      (total, target) => total + target.gain,
      0,
    );
    if (!bestPlan || gain > bestPlan.gain)
      bestPlan = {
        program: program.name,
        gain,
        cells,
        selected: new Set([...cells.values()].map((target) => target.index)),
      };
  }
  return bestPlan;
}
function drawField(plan) {
  if (!canvas) return;
  const width = canvas.getBoundingClientRect().width;
  if (!width) return;
  const height = width * 0.75;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(pixelRatio, pixelRatio);
  const size = Math.min(width * 0.7, height * 0.83);
  const left = (width - size) / 2,
    top = (height - size) / 2;
  const cellSize = size / 4;
  skyStars.forEach((star) => {
    ctx.beginPath();
    ctx.arc(star.x * width, star.y * height, star.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(148,177,231,${star.opacity})`;
    ctx.fill();
  });
  ctx.strokeStyle = "rgba(89,124,193,0.12)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath();
    ctx.ellipse(
      width / 2,
      height / 2,
      size * (0.5 + i * 0.12),
      size * (0.5 + i * 0.12),
      0,
      0,
      Math.PI * 2,
    );
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(5,12,30,.38)";
  ctx.fillRect(left, top, size, size);
  ctx.font = `${Math.max(7, width / 78)}px ui-monospace, monospace`;
  ctx.textAlign = "center";
  for (let i = 0; i < 4; i++) {
    ctx.fillStyle = "#8295bc";
    ctx.fillText(`C${i + 1}`, left + (i + 0.5) * cellSize, top - 12);
    ctx.fillText(`R${i + 1}`, left - 18, top + (i + 0.5) * cellSize + 3);
  }
  plan.cells.forEach((selection, cell) => {
    const x = left + (cell % 4) * cellSize,
      y = top + Math.floor(cell / 4) * cellSize;
    ctx.fillStyle = "rgba(119,162,255,.045)";
    ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
  });
  ctx.strokeStyle = "rgba(119,162,255,.42)";
  for (let i = 0; i <= 4; i++) {
    ctx.beginPath();
    ctx.moveTo(left + i * cellSize, top);
    ctx.lineTo(left + i * cellSize, top + size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(left, top + i * cellSize);
    ctx.lineTo(left + size, top + i * cellSize);
    ctx.stroke();
  }
  ctx.strokeStyle = "#8db1ff";
  const corner = 10;
  [
    [left, top, 1, 1],
    [left + size, top, -1, 1],
    [left, top + size, 1, -1],
    [left + size, top + size, -1, -1],
  ].forEach(([x, y, dx, dy]) => {
    ctx.beginPath();
    ctx.moveTo(x + dx * corner, y - dy * 5);
    ctx.lineTo(x - dx * 5, y - dy * 5);
    ctx.lineTo(x - dx * 5, y + dy * corner);
    ctx.stroke();
  });
  targets.forEach((target, index) => {
    const x = left + target.x * size,
      y = top + target.y * size;
    const selected = plan.selected.has(index),
      radius = Math.max(1.7, width / 190) * (selected ? 1.15 : 0.75);
    if (selected) {
      const glow = ctx.createRadialGradient(x, y, 0, x, y, radius * 5);
      glow.addColorStop(0, "rgba(119,162,255,.4)");
      glow.addColorStop(1, "rgba(119,162,255,0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(x, y, radius * 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(119,162,255,.65)";
      ctx.beginPath();
      ctx.arc(x, y, radius * 2.8, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = selected
      ? "#deebff"
      : target.best >= 1.45
        ? "#50607b"
        : "#a8bcdf";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = "#8295bc";
  ctx.font = `${Math.max(7, width / 90)}px ui-monospace, monospace`;
  ctx.fillText("FIXED FIELD / ONE TARGET PER FIBRE", width / 2, height - 12);
}
function updateDemo() {
  const seconds = Number(durationInput.value),
    plan = evaluate(seconds);
  document.querySelector("#duration-label").textContent =
    `${seconds.toLocaleString("zh-CN")} 秒`;
  document.querySelector("#demo-fibres").textContent =
    `${plan.cells.size} / 16`;
  document.querySelector("#demo-gain").textContent = `+${plan.gain.toFixed(2)}`;
  document.querySelector("#demo-program").textContent = plan.program;
  const explanation =
    weather >= 0.9
      ? "晴好天空让目标更快达到完成度；继续加长曝光，新增收益会逐渐变小。"
      : weather >= 0.6
        ? "普通天空下，要在新增收益与占用时间之间取舍。每个目标只计最好的一次曝光。"
        : "较差天空需要更多时间。程序可能切换到 BRIGHT 或 BACKUP；低于已有成绩的曝光不会带来新增收益。";
  document.querySelector("#demo-explanation").textContent = explanation;
  drawField(plan);
}
weatherButtons.forEach((button) =>
  button.addEventListener("click", () => {
    weather = Number(button.dataset.weather);
    weatherButtons.forEach((item) =>
      item.setAttribute("aria-pressed", String(item === button)),
    );
    updateDemo();
  }),
);
durationInput.addEventListener("input", updateDemo);
document.querySelector("#auto-duration").addEventListener("click", () => {
  const durations = [
    90, 120, 180, 240, 300, 360, 450, 540, 660, 780, 900, 1080, 1260, 1500,
    1800, 2100, 2400, 2700, 3000, 3300, 3600,
  ];
  // A fixed illustrative time price; this is not the real season estimator.
  let bestDuration = 90,
    bestNet = -Infinity;
  for (const seconds of durations) {
    const net = evaluate(seconds).gain - 0.006 * seconds;
    if (net > bestNet) {
      bestNet = net;
      bestDuration = seconds;
    }
  }
  durationInput.value = bestDuration;
  updateDemo();
  document.querySelector("#demo-explanation").textContent =
    `示例以“新增收益 − 0.006 × 秒数”比较候选时长，选择 ${bestDuration.toLocaleString("zh-CN")} 秒。实际时间价格会随赛季与可观测目标变化。`;
});
if (typeof ResizeObserver !== "undefined")
  new ResizeObserver(() =>
    drawField(evaluate(Number(durationInput.value))),
  ).observe(canvas);
else
  window.addEventListener("resize", () =>
    drawField(evaluate(Number(durationInput.value))),
  );
updateDemo();

// Decorative sky and instrument coordinates use their own deterministic catalogue.
// No animation loop or live measurements: both canvases redraw only on resize.
const skyBackground = document.querySelector("#sky-background");
const heroMap = document.querySelector("#hero-map");
function normal() {
  return (random() + random() + random() + random() - 2) / 2;
}
const backgroundPoints = Array.from({ length: 3400 }, (_, i) => {
  const radius = Math.sqrt(random()) * 0.53;
  const angle = ((i % 3) * Math.PI * 2) / 3 + radius * 12 + normal() * 1.25;
  return {
    x: 0.75 + Math.cos(angle) * radius * 0.75,
    y: 0.5 + Math.sin(angle) * radius,
    r: 0.3 + random() * 1.4,
    opacity: 0.08 + random() * 0.55,
    pale: random() > 0.87,
  };
});
const mapPoints = Array.from({ length: 530 }, (_, i) => {
  const r = Math.sqrt(random()) * 0.95,
    a = random() * Math.PI * 2;
  return {
    x: Math.cos(a) * r,
    y: Math.sin(a) * r,
    radius: 0.5 + random() * 1.15,
    required: i % 37 === 0,
  };
});
function canvasContext(element) {
  if (!element) return null;
  const bounds = element.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return null;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  element.width = Math.round(bounds.width * ratio);
  element.height = Math.round(bounds.height * ratio);
  const ctx = element.getContext("2d");
  if (!ctx) return null;
  ctx.scale(ratio, ratio);
  return { ctx, width: bounds.width, height: bounds.height };
}
function drawSkyBackground() {
  const surface = canvasContext(skyBackground);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const glow = ctx.createRadialGradient(
    width * 0.76,
    height * 0.47,
    0,
    width * 0.76,
    height * 0.47,
    width * 0.5,
  );
  glow.addColorStop(0, "rgba(30,67,170,.24)");
  glow.addColorStop(0.5, "rgba(25,39,101,.12)");
  glow.addColorStop(1, "rgba(5,7,13,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);
  backgroundPoints.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, point.r, 0, Math.PI * 2);
    ctx.fillStyle = point.pale
      ? `rgba(202,218,255,${point.opacity})`
      : `rgba(70,114,239,${point.opacity})`;
    ctx.fill();
  });
}
function drawHeroMap() {
  const surface = canvasContext(heroMap);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const cx = width / 2,
    cy = height / 2,
    radius = Math.min(width * 0.42, height * 0.44);
  ctx.strokeStyle = "rgba(104,135,197,.2)";
  ctx.lineWidth = 0.65;
  for (let ring = 1; ring <= 4; ring++) {
    ctx.beginPath();
    ctx.arc(cx, cy, (radius * ring) / 4, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (let line = 0; line < 12; line++) {
    const angle = (line * Math.PI) / 6;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(113,151,229,.4)";
  for (let tick = 0; tick < 72; tick++) {
    const angle = (tick * Math.PI) / 36,
      inside = radius - (tick % 6 === 0 ? 6 : 2.5);
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * inside, cy + Math.sin(angle) * inside);
    ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    ctx.stroke();
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `${Math.max(7, width / 65)}px "IBM Plex Mono",monospace`;
  ctx.fillStyle = "#90a7cf";
  [
    ["N", 0, -1],
    ["E", 1, 0],
    ["S", 0, 1],
    ["W", -1, 0],
  ].forEach(([name, x, y]) =>
    ctx.fillText(name, cx + x * (radius + 13), cy + y * (radius + 13)),
  );
  mapPoints.forEach((point) => {
    const x = cx + point.x * radius,
      y = cy + point.y * radius,
      r = point.radius * Math.max(0.8, width / 570);
    if (point.required) {
      ctx.fillStyle = "rgba(255,208,120,.1)";
      ctx.beginPath();
      ctx.arc(x, y, r * 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = point.required ? "#ffd078" : "rgba(126,166,244,.65)";
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  });
  const fieldSize = radius * 0.45;
  ctx.save();
  ctx.translate(cx + radius * 0.26, cy - radius * 0.25);
  ctx.rotate(-Math.PI / 10);
  ctx.fillStyle = "rgba(61,101,230,.12)";
  ctx.fillRect(-fieldSize / 2, -fieldSize / 2, fieldSize, fieldSize);
  ctx.strokeStyle = "rgba(209,229,255,.85)";
  ctx.lineWidth = 0.7;
  ctx.strokeRect(-fieldSize / 2, -fieldSize / 2, fieldSize, fieldSize);
  ctx.strokeStyle = "rgba(177,208,255,.4)";
  for (let i = 1; i < 4; i++) {
    const v = -fieldSize / 2 + (i * fieldSize) / 4;
    ctx.beginPath();
    ctx.moveTo(v, -fieldSize / 2);
    ctx.lineTo(v, fieldSize / 2);
    ctx.moveTo(-fieldSize / 2, v);
    ctx.lineTo(fieldSize / 2, v);
    ctx.stroke();
  }
  for (let row = 0; row < 4; row++)
    for (let col = 0; col < 4; col++) {
      ctx.fillStyle = "#e8f1ff";
      ctx.beginPath();
      ctx.arc(
        -fieldSize / 2 + ((col + 0.5) * fieldSize) / 4,
        -fieldSize / 2 + ((row + 0.5) * fieldSize) / 4,
        Math.max(1, width / 320),
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
  ctx.restore();
  ctx.strokeStyle = "rgba(181,209,255,.6)";
  ctx.beginPath();
  ctx.moveTo(cx - 6, cy);
  ctx.lineTo(cx + 6, cy);
  ctx.moveTo(cx, cy - 6);
  ctx.lineTo(cx, cy + 6);
  ctx.stroke();
}
function redrawSky() {
  drawSkyBackground();
  drawHeroMap();
}
if (typeof ResizeObserver !== "undefined") {
  const skyResize = new ResizeObserver(redrawSky);
  skyResize.observe(skyBackground);
  skyResize.observe(heroMap);
} else window.addEventListener("resize", redrawSky);
redrawSky();
function updateScrollProgress() {
  const extent = document.documentElement.scrollHeight - innerHeight;
  document.querySelector("#scroll-progress").style.transform =
    `scaleX(${extent > 0 ? scrollY / extent : 0})`;
}
window.addEventListener("scroll", updateScrollProgress, { passive: true });
window.addEventListener("resize", updateScrollProgress);
updateScrollProgress();

// WAI-ARIA tabs: keyboard navigation, a single tab stop and paired panels.
const tabs = [...document.querySelectorAll('[role="tab"]')];
function activateTab(tab, focus = false) {
  tabs.forEach((item) => {
    const active = item === tab;
    item.setAttribute("aria-selected", String(active));
    item.tabIndex = active ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden =
      !active;
  });
  if (focus) tab.focus();
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateTab(tab));
  tab.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft")
      next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    activateTab(tabs[next], true);
  });
});

const menuButton = document.querySelector(".menu-toggle"),
  navigation = document.querySelector("#navigation");
function closeMenu() {
  menuButton.setAttribute("aria-expanded", "false");
  navigation.classList.remove("is-open");
}
menuButton.addEventListener("click", () => {
  const opened = menuButton.getAttribute("aria-expanded") !== "true";
  menuButton.setAttribute("aria-expanded", String(opened));
  navigation.classList.toggle("is-open", opened);
});
navigation.addEventListener("click", (event) => {
  if (event.target.closest("a")) closeMenu();
});
document.addEventListener("keydown", (event) => {
  if (
    event.key === "Escape" &&
    menuButton.getAttribute("aria-expanded") === "true"
  ) {
    closeMenu();
    menuButton.focus();
  }
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".site-header")) closeMenu();
});

document.querySelector("#copy-clone").addEventListener("click", async () => {
  const text = document.querySelector("#clone-command").textContent;
  const status = document.querySelector("#copy-status");
  try {
    if (!navigator.clipboard?.writeText)
      throw new Error("clipboard unavailable");
    await navigator.clipboard.writeText(text);
    status.textContent = "命令已复制";
  } catch {
    const selection = window.getSelection(),
      range = document.createRange();
    range.selectNodeContents(document.querySelector("#clone-command"));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "命令已选中，可手动复制";
  }
});
