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
    ctx.fillStyle = `rgba(161,188,193,${star.opacity})`;
    ctx.fill();
  });
  ctx.strokeStyle = "rgba(102,143,147,0.11)";
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
  ctx.fillStyle = "rgba(7,20,24,.38)";
  ctx.fillRect(left, top, size, size);
  ctx.font = `${Math.max(7, width / 78)}px ui-monospace, monospace`;
  ctx.textAlign = "center";
  for (let i = 0; i < 4; i++) {
    ctx.fillStyle = "#69848a";
    ctx.fillText(`C${i + 1}`, left + (i + 0.5) * cellSize, top - 12);
    ctx.fillText(`R${i + 1}`, left - 18, top + (i + 0.5) * cellSize + 3);
  }
  plan.cells.forEach((selection, cell) => {
    const x = left + (cell % 4) * cellSize,
      y = top + Math.floor(cell / 4) * cellSize;
    ctx.fillStyle = "rgba(121,223,184,.045)";
    ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
  });
  ctx.strokeStyle = "rgba(121,223,184,.42)";
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
  ctx.strokeStyle = "#79dfb8";
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
      glow.addColorStop(0, "rgba(121,223,184,.4)");
      glow.addColorStop(1, "rgba(121,223,184,0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(x, y, radius * 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(121,223,184,.65)";
      ctx.beginPath();
      ctx.arc(x, y, radius * 2.8, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = selected
      ? "#a6f3d3"
      : target.best >= 1.45
        ? "#4b6066"
        : "#9db1b9";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = "#6e8b90";
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

const heroStars = document.querySelector("#hero-stars");
for (let i = 0; i < 100; i++) {
  const star = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  star.setAttribute("cx", String(70 + random() * 490));
  star.setAttribute("cy", String(70 + random() * 410));
  star.setAttribute("r", String(0.5 + random() * 1.3));
  star.setAttribute("opacity", String(0.2 + random() * 0.55));
  heroStars.appendChild(star);
}

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
