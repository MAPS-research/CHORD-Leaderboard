// CHORD Leaderboard page: renders data/leaderboard.json (built by `python -m leaderboard_site.build`).
const GROUP_LABEL = { ar: "AR", discrete: "Discrete diffusion/flow", continuous: "Continuous diffusion/flow" };
const fmtDate = d => new Date(d + "T00:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const score = m => `${m.mean.toFixed(2)} <small>(±${m.std.toFixed(2)})</small>`;
const links = m => (m.paper ? `<a href="${esc(m.paper)}">Paper</a>` : "") + `<a href="${esc(m.code)}">GitHub</a>`;
const tag = m => `<span class="tag ${esc(m.group)}">${GROUP_LABEL[m.group] ?? esc(m.group)}</span>`;
const cmp = {
  mean: (a, b) => a.mean - b.mean,
  params: (a, b) => a.params_value - b.params_value,
  released: (a, b) => a.released.localeCompare(b.released),
};

let data = null, sort = { key: "mean", dir: 1 }, group = "all";

const humanRow = h => `<tr class="human"><td class="rank">-</td><td>${esc(h.name)}</td><td class="center score">${score(h)}</td>
  <td class="center col-params">-</td><td class="col-date">-</td><td class="col-links">-</td></tr>`;
const modelRow = (m, rank) => `<tr><td class="rank">${rank}</td>
  <td>${esc(m.name)}${tag(m)}<div class="meta">${esc(m.params)} · ${fmtDate(m.released)}</div><div class="meta links">${links(m)}</div></td>
  <td class="center score">${score(m)}</td>
  <td class="center col-params">${esc(m.params)}</td><td class="col-date">${fmtDate(m.released)}</td>
  <td class="links col-links">${links(m)}</td></tr>`;

const pendingRow = m => `<tr>
  <td>${esc(m.name)}${tag(m)}<div class="meta">${esc(m.params ?? "-")} · ${m.released ? fmtDate(m.released) : "-"}</div><div class="meta links">${links(m)}</div></td>
  <td class="center col-params">${esc(m.params ?? "-")}</td><td class="col-date">${m.released ? fmtDate(m.released) : "-"}</td>
  <td class="links col-links">${links(m)}</td></tr>`;

function renderPending() {
  const rows = data.pending || [];
  document.getElementById("pending-rows").innerHTML = rows.length
    ? rows.map(pendingRow).join("")
    : '<tr><td class="status" colspan="4">No generators are waiting for evaluation.</td></tr>';
}

const byName = (a, b) => a.name.localeCompare(b.name, "en", { numeric: true });
const ckptRow = m => `<tr><td>${esc(m.name)}${tag(m)}</td>
  <td class="links ckpt"><a href="${esc(m.checkpoint.url)}">${esc(m.checkpoint.label)}</a></td></tr>`;

function renderCheckpoints() {
  const rows = [...data.generators, ...(data.pending || [])].sort(byName);
  document.getElementById("ckpt-rows").innerHTML = rows.map(ckptRow).join("");
}

function render() {
  const pool = data.generators.filter(m => group === "all" || m.group === group);
  const rankOf = new Map([...pool].sort(cmp.mean).map((m, i) => [m.id, i + 1]));  // rank within the selected type
  const rows = [...pool].sort((a, b) => sort.dir * cmp[sort.key](a, b));
  document.getElementById("rows").innerHTML = humanRow(data.human) + rows.map(m => modelRow(m, rankOf.get(m.id))).join("");
  document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-selected", String(t.dataset.group === group)));
  document.querySelectorAll("th button").forEach(b =>
    b.setAttribute("aria-sort", b.dataset.key === sort.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"));
}

document.querySelectorAll("th button").forEach(b => b.addEventListener("click", () => {
  const key = b.dataset.key;
  sort = sort.key === key ? { key, dir: -sort.dir } : { key, dir: key === "mean" ? 1 : -1 };
  if (data) render();
}));
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => { group = t.dataset.group; if (data) render(); }));
document.getElementById("settings-toggle").addEventListener("click", e => {
  const info = document.getElementById("settings-info");
  info.hidden = !info.hidden;
  e.currentTarget.setAttribute("aria-expanded", String(!info.hidden));
});

fetch("data/leaderboard.json", { cache: "no-cache" })
  .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then(d => {
    data = d;
    const t = document.getElementById("last-modified");
    t.dateTime = d.last_modified;
    t.textContent = fmtDate(d.last_modified);
    render();
    renderPending();
    renderCheckpoints();
  })
  .catch(() => {
    document.getElementById("rows").innerHTML = '<tr><td class="status" colspan="6">Could not load the leaderboard data.</td></tr>';
    document.getElementById("pending-rows").innerHTML = '<tr><td class="status" colspan="4">Could not load the leaderboard data.</td></tr>';
    document.getElementById("ckpt-rows").innerHTML = '<tr><td class="status" colspan="2">Could not load the leaderboard data.</td></tr>';
  });
