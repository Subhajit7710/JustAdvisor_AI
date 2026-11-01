/* Nyaya Moot Court - front end (no build step, no libraries). */
const $view = document.getElementById("view");
const $engine = document.getElementById("engine");

const state = {
  status: null,        // engines + limits
  cases: [],
  session: null,
  role: "both",        // which lawyer(s) use this device: both | A | B
  draft: { A: "", B: "" },
  assist: { A: null, B: null },
  assistBusy: false,
  busy: false,
  error: null,
  pollTimer: null,
};

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
  return data;
}

function store(key, value) { try { localStorage.setItem(key, value); } catch (_) {} }
function load(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }

const lawyerName = (side) => `Lawyer ${side}`;
function roleOf(c, side) { return side === "A" ? c.lawyer_a_role : c.lawyer_b_role; }
function shortRole(c, side) {
  const r = roleOf(c, side).replace(/^the /, "");
  return r.split(" (")[0];
}

// ------------------------------------------------------------------ engine status (header)
async function loadStatus() {
  try {
    state.status = await api("/api/status");
    const s = state.status;
    const judge = s.judge_ready ? `Judge: trained model` : `Judge: demo mode`;
    const asst = s.assistant_ready ? `Assistant: AI model` : `Assistant: basic suggestions`;
    $engine.innerHTML = `<span class="${s.judge_ready ? "" : "demo"}">${esc(judge)}</span><span class="${s.assistant_ready ? "" : "demo"}">${esc(asst)}</span>` +
      (s.legalbert_ready ? `<span>Structuring: LegalBERT</span>` : "");
  } catch (e) {
    $engine.textContent = "Server not reachable";
  }
}

// ------------------------------------------------------------------ routing
function route() {
  clearTimeout(state.pollTimer);
  const m = location.hash.match(/^#\/h\/([A-Za-z0-9]+)/);
  if (m) openHearing(m[1].toUpperCase());
  else renderHome();
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ home: cause list
async function renderHome() {
  state.session = null;
  if (!state.cases.length) {
    try { state.cases = await api("/api/cases"); } catch (e) { $view.innerHTML = `<p class="notice err">${esc(e.message)}</p>`; return; }
  }
  const today = new Date().toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" });
  $view.innerHTML = `
    <section class="home-head">
      <h1>Argue a case before an AI judge</h1>
      <p>Two lawyers take turns presenting arguments, with an AI assistant to help each side. After the arguments close, the judge weighs every submission issue by issue and delivers a reasoned verdict.</p>
    </section>
    <section class="causelist" aria-labelledby="cl">
      <div class="causelist-title"><h2 id="cl">Cause list</h2><small>${esc(today)} &middot; ${state.cases.length} matters</small></div>
      ${state.cases.map((c, i) => `
        <div class="item">
          <div class="item-no" aria-hidden="true">${i + 1}</div>
          <div>
            <h3>${esc(c.title)}</h3>
            <p class="law"><span class="nature ${c.nature}">${c.nature === "criminal" ? "Criminal" : "Civil"}</span>${esc(c.law)}</p>
          </div>
          <button class="btn" data-case="${esc(c.id)}">Open hearing</button>
        </div>`).join("")}
    </section>
    <form class="join" id="join">
      <label for="code">Joining from another device? Enter the hearing code</label>
      <input id="code" maxlength="6" autocomplete="off" placeholder="A1B2C3">
      <button class="btn ghost" type="submit">Join hearing</button>
    </form>`;
  $view.querySelectorAll("[data-case]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    try {
      const s = await api("/api/sessions", { method: "POST", body: JSON.stringify({ case_id: b.dataset.case }) });
      location.hash = `#/h/${s.id}`;
    } catch (e) { b.disabled = false; alertBox(e.message); }
  }));
  document.getElementById("join").addEventListener("submit", (e) => {
    e.preventDefault();
    const code = document.getElementById("code").value.trim().toUpperCase();
    if (code) location.hash = `#/h/${code}`;
  });
}

function alertBox(msg) {
  const n = document.createElement("p");
  n.className = "notice err";
  n.textContent = msg;
  $view.prepend(n);
}

// ------------------------------------------------------------------ hearing
async function openHearing(code) {
  try {
    state.session = await api(`/api/sessions/${code}`);
  } catch (e) {
    $view.innerHTML = `<p class="notice err">${esc(e.message)}</p><p><a href="#/">Back to the cause list</a></p>`;
    return;
  }
  state.role = load(`role-${code}`) || "both";
  state.draft = { A: load(`draft-${code}-A`) || "", B: load(`draft-${code}-B`) || "" };
  state.assist = { A: null, B: null };
  state.error = null;
  renderHearing();
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  const s = state.session;
  if (!s || s.status === "decided") return;
  const fast = s.status === "deliberating" || (state.role !== "both" && s.next_speaker && s.next_speaker !== state.role);
  state.pollTimer = setTimeout(async () => {
    try {
      const fresh = await api(`/api/sessions/${s.id}`);
      const changed = JSON.stringify([fresh.status, fresh.turns.length, fresh.judgment && fresh.judgment.text.length, fresh.error]) !==
        JSON.stringify([s.status, s.turns.length, s.judgment && s.judgment.text.length, s.error]);
      state.session = fresh;
      if (changed) renderHearing({ keepFocus: true });
    } catch (_) { /* keep polling */ }
    schedulePoll();
  }, fast ? 1200 : 4000);
}

function issueMarks(s) {
  const p = s.judgment && s.judgment.parsed;
  if (!p) return {};
  const m = {};
  p.issues.forEach((i) => { m[i.n] = i.finding; });
  return m;
}

function renderHearing(opts = {}) {
  const s = state.session;
  const c = s.case;
  const active = document.activeElement && document.activeElement.id;
  const marks = issueMarks(s);

  const rounds = [];
  s.turns.forEach((t) => { (rounds[t.round - 1] = rounds[t.round - 1] || []).push(t); });

  const meter = Array.from({ length: s.limits.max_rounds }, (_, i) => {
    const cls = i < s.completed_rounds ? "done" : (i === s.completed_rounds && s.status === "arguing" ? "now" : "");
    return `<i class="${cls}"></i>`;
  }).join("");

  $view.innerHTML = `
    <div class="hearing-head">
      <div>
        <h1>${esc(c.title)}</h1>
        <div class="sub">${esc(c.law)}</div>
      </div>
      <div class="code">Hearing code <b>${esc(s.id)}</b></div>
    </div>
    <div class="layout">
      <aside class="casefile" aria-label="Case file">
        <h2>Facts</h2>
        <p class="facts">${esc(c.facts)}</p>
        <ul class="parties">
          <li class="pa"><b>Lawyer A</b> appears for ${esc(c.lawyer_a_role)}</li>
          <li class="pb"><b>Lawyer B</b> appears for ${esc(c.lawyer_b_role)}</li>
        </ul>
        <h2>Issues to be decided</h2>
        <ol class="issues">
          ${c.issues.map((x, i) => {
            const f = marks[i + 1];
            const cls = f === "PROVED" ? "proved" : f === "NOT PROVED" ? "notproved" : "";
            return `<li class="${cls}">${esc(x.replace(/^Whether /, "Whether "))}${f ? ` <b>${f === "PROVED" ? "Proved" : "Not proved"}</b>` : ""}</li>`;
          }).join("")}
        </ol>
        <p class="burden">Lawyer A must prove every issue (${c.nature === "criminal" ? "beyond reasonable doubt" : "on the balance of probabilities"}). Lawyer B wins if any one issue fails.</p>
      </aside>

      <section aria-label="Proceedings">
        <div class="record">
          <div class="record-head">
            <h2>Record of proceedings</h2>
            <div class="rounds-meter" aria-label="${s.completed_rounds} of ${s.limits.max_rounds} rounds complete">${meter}<span>&nbsp;${s.completed_rounds}/${s.limits.max_rounds} rounds</span></div>
          </div>
          ${s.turns.length ? rounds.map((r, i) => `
            <div class="round">
              <div class="round-label">Round ${i + 1}</div>
              ${r.map((t) => `
                <div class="entry ${t.speaker}">
                  <div class="who">${lawyerName(t.speaker)}<small>${esc(shortRole(c, t.speaker))}</small></div>
                  <p>${renderTagged(t)}</p>
                </div>`).join("")}
            </div>`).join("") : `<p class="empty-record">No arguments yet. Lawyer A opens the case.</p>`}
        </div>
        ${renderStructure(s)}
        ${s.error ? `<p class="notice err">${esc(s.error)}</p>` : ""}
        ${state.error ? `<p class="notice err">${esc(state.error)}</p>` : ""}
        ${renderStage(s)}
      </section>
    </div>`;

  bindHearing();
  if (opts.keepFocus && active) {
    const el = document.getElementById(active);
    if (el) { el.focus(); if (el.setSelectionRange && el.value) el.setSelectionRange(el.value.length, el.value.length); }
  }
}

// LegalBERT: each sentence tagged Claim / Evidence / LegalRule / Rebuttal / Conclusion
const ROLE_NAME = { Claim: "Claim", Evidence: "Evidence", LegalRule: "Legal rule", Rebuttal: "Rebuttal", Conclusion: "Conclusion" };
function renderTagged(t) {
  if (!t.structure || !t.structure.length) return esc(t.text);
  return t.structure.map((x) =>
    `<span class="tagged"><span class="tag role-${esc(x.role)}" title="LegalBERT: ${esc(ROLE_NAME[x.role] || x.role)} (${Math.round(x.confidence * 100)}% confident)">${esc(ROLE_NAME[x.role] || x.role)}</span>${esc(x.sentence)}</span>`
  ).join(" ");
}

const GROUPS = [["claims", "Claims"], ["evidence", "Evidence"], ["laws", "Legal rules"], ["rebuttals", "Rebuttals"], ["conclusions", "Conclusions"]];
function renderStructure(s) {
  if (!s.structured) return "";
  const col = (side) => {
    const g = s.structured[`Lawyer_${side}`];
    const filled = GROUPS.filter(([k]) => g[k].length);
    return `
      <div class="struct-col ${side}">
        <h3>${lawyerName(side)} <small>${esc(shortRole(s.case, side))}</small></h3>
        ${filled.length ? filled.map(([k, label]) => `
          <h4>${label} <span>${g[k].length}</span></h4>
          <ul>${g[k].map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`).join("") : `<p class="muted">No arguments yet.</p>`}
      </div>`;
  };
  return `
    <details class="structure" id="structure" ${state.structOpen ? "open" : ""}>
      <summary>Argument structure <span class="muted">sentences grouped by LegalBERT</span></summary>
      <div class="struct-grid">${col("A")}${col("B")}</div>
    </details>`;
}

function renderStage(s) {
  if (s.status === "deliberating") return renderDeliberation(s);
  if (s.status === "decided") return renderJudgment(s);

  const c = s.case;
  const side = s.next_speaker;
  const roleButtons = `
    <div class="role-pick" role="group" aria-label="Who uses this device">
      This device is for:
      ${[["both", "Both lawyers"], ["A", "Lawyer A only"], ["B", "Lawyer B only"]].map(([v, l]) =>
        `<button type="button" data-role="${v}" aria-pressed="${state.role === v}">${l}</button>`).join("")}
    </div>`;

  let composer = "";
  if (side && (state.role === "both" || state.role === side)) {
    const text = state.draft[side];
    const max = s.limits.max_chars;
    const a = state.assist[side];
    composer = `
      <div class="composer ${side}">
        <div class="composer-head">
          <strong>${lawyerName(side)}, your turn</strong>
          <span class="count">Round ${s.round} of ${s.limits.max_rounds} &middot; for ${esc(shortRole(c, side))}</span>
        </div>
        <label class="sr-only" for="draft">Your argument</label>
        <textarea id="draft" maxlength="${max + 200}" placeholder="${side === "A"
          ? "State your argument and the evidence it rests on, e.g. documents, records or witness testimony."
          : "Respond to Lawyer A. Rebut their evidence or show that an issue is not proved."}">${esc(text)}</textarea>
        <div class="composer-foot">
          <span class="count ${text.length > max ? "over" : ""}" id="count">${text.length} / ${max} characters</span>
          <div class="actions">
            <button class="btn ghost" id="ask" ${state.assistBusy || s.assist_left[side] <= 0 ? "disabled" : ""}>${state.assistBusy ? "Assistant is thinking…" : `Ask assistant (${s.assist_left[side]} left)`}</button>
            <button class="btn ${side.toLowerCase()}" id="submit" ${state.busy ? "disabled" : ""}>Submit argument</button>
          </div>
        </div>
        ${a ? renderAssist(a, side) : ""}
      </div>`;
  } else if (side) {
    composer = `<div class="waiting">Waiting for ${lawyerName(side)} to argue on their device. This page updates automatically.</div>`;
  }

  const closed = !side;
  const verdictBar = (s.can_request_verdict || closed) ? `
    <div class="verdict-bar">
      <p>${closed ? `All ${s.limits.max_rounds} rounds are complete. Arguments are closed.`
        : `${s.completed_rounds} rounds complete. You can continue arguing, or close arguments and ask the judge for a verdict.`}</p>
      <button class="btn bench-btn" id="verdict" ${state.busy ? "disabled" : ""}>Close arguments and ask for the verdict</button>
    </div>` : "";

  return composer + verdictBar + roleButtons;
}

function renderAssist(a, side) {
  return `
    <div class="assist" aria-live="polite">
      <h3>Suggestions for ${lawyerName(side)}</h3>
      <p class="hint">${a.engine === "model" ? "From the AI assistant, using the legal sources below." : "Basic suggestions (the AI assistant model isn't installed)."} Edit them into your own words before submitting.${a.note ? " " + esc(a.note) : ""}</p>
      ${a.suggestions.map((x, i) => `
        <div class="sugg">
          <div class="sugg-top">${x.issue ? `<span>Issue ${x.issue}</span>` : ""}<span class="strength ${esc(x.strength)}">${esc(x.strength)}</span>${x.sources && x.sources.length ? `<span>Cites ${esc(x.sources.join(", "))}</span>` : ""}</div>
          <p>${esc(x.argument)}</p>
          ${x.why ? `<div class="why">${esc(x.why)}</div>` : ""}
          <button class="btn ghost small" data-use="${i}">Use this</button>
        </div>`).join("")}
      ${a.sources && a.sources.length ? `
        <details class="sources"><summary>Legal sources (${a.sources.length})</summary><ul>
          ${a.sources.map((s) => `<li><b>${esc(s.id)}</b> ${esc(s.title)}: ${esc(s.text)}</li>`).join("")}
        </ul></details>` : ""}
    </div>`;
}

function renderDeliberation(s) {
  const j = s.judgment || {};
  const secs = j.started ? Math.max(0, Math.round(Date.now() / 1000 - j.started)) : 0;
  return `
    <div class="deliberating" aria-live="polite">
      <h2>The judge is considering the arguments<span class="gavel-dots"><span>.</span><span>.</span><span>.</span></span></h2>
      <div class="sub">${j.engine === "model" ? "On a computer without a graphics card this usually takes 1 to 3 minutes." : "Demo judge."} ${secs ? `${secs}s so far.` : ""}</div>
      <div class="stream" id="stream">${esc(j.text || "")}</div>
    </div>`;
}

function renderJudgment(s) {
  const j = s.judgment;
  const p = j.parsed || { issues: [], no_weight: [], verdict: null };
  const c = s.case;
  const winnerText = p.winner ? `In favour of ${lawyerName(p.winner)} (${esc(shortRole(c, p.winner))})` : "";
  const verdictLabel = { "GUILTY": "Guilty", "NOT GUILTY": "Not guilty", "CLAIM ALLOWED": "Claim allowed", "CLAIM DISMISSED": "Claim dismissed" }[p.verdict] || "No clear verdict";
  return `
    <article class="judgment" aria-labelledby="jh">
      <div class="stamp" aria-hidden="true">${esc(verdictLabel)}<small>${s.completed_rounds} rounds heard</small></div>
      <h2 id="jh">Judgment</h2>
      <div class="meta">${winnerText}${j.seconds != null ? ` &middot; delivered in ${j.seconds}s` : ""}</div>
      ${j.engine === "demo" ? `<div class="demo-note">This verdict comes from the rule-based demo judge, not the trained AI model.</div>` : ""}
      ${!p.complete ? `<div class="demo-note">The judgment text was incomplete, so some findings may be missing. The full text is below.</div>` : ""}
      ${p.issues.map((i) => {
        const ok = i.finding === "PROVED";
        return `
        <div class="finding ${ok ? "proved" : "notproved"}">
          <div class="mark" aria-hidden="true">${ok ? "✓" : "✗"}</div>
          <div>
            <h3>Issue ${i.n}: ${esc(i.issue)}</h3>
            <div class="res">${ok ? "Proved" : "Not proved"}. <span style="font-weight:400;color:var(--ink)">${esc(i.reason)}</span></div>
            <ul>${i.submissions.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
          </div>
        </div>`;
      }).join("")}
      ${p.no_weight && p.no_weight.length ? `<div class="noweight"><b>Submissions given no weight</b><ul>${p.no_weight.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>` : ""}
      ${p.order ? `<p class="order">${esc(p.order)}</p>` : ""}
      <details class="fulltext"><summary>Full judgment text</summary><pre>${esc(j.text)}</pre></details>
      <div class="judgment-actions">
        <button class="btn" onclick="location.hash='#/'">Start a new hearing</button>
        <button class="btn ghost" onclick="window.print()">Print judgment</button>
      </div>
    </article>`;
}

// ------------------------------------------------------------------ events
function bindHearing() {
  const det = document.getElementById("structure");
  if (det) det.addEventListener("toggle", () => { state.structOpen = det.open; });
  const s = state.session;
  const side = s.next_speaker;
  const ta = document.getElementById("draft");
  if (ta) {
    ta.addEventListener("input", () => {
      state.draft[side] = ta.value;
      store(`draft-${s.id}-${side}`, ta.value);
      const cnt = document.getElementById("count");
      cnt.textContent = `${ta.value.length} / ${s.limits.max_chars} characters`;
      cnt.classList.toggle("over", ta.value.length > s.limits.max_chars);
    });
  }
  const submit = document.getElementById("submit");
  if (submit) submit.addEventListener("click", async () => {
    state.busy = true; state.error = null; renderHearing();
    try {
      state.session = await api(`/api/sessions/${s.id}/turns`, { method: "POST", body: JSON.stringify({ speaker: side, text: state.draft[side] }) });
      state.draft[side] = ""; store(`draft-${s.id}-${side}`, ""); state.assist[side] = null;
    } catch (e) { state.error = e.message; }
    state.busy = false; renderHearing(); schedulePoll();
    const next = document.getElementById("draft"); if (next) next.focus();
  });
  const ask = document.getElementById("ask");
  if (ask) ask.addEventListener("click", async () => {
    state.assistBusy = true; state.error = null; renderHearing();
    try {
      const res = await api(`/api/sessions/${s.id}/assist`, { method: "POST", body: JSON.stringify({ speaker: side, draft: state.draft[side] }) });
      state.assist[side] = res;
      state.session.assist_left[side] = res.assist_left;
    } catch (e) { state.error = e.message; }
    state.assistBusy = false; renderHearing();
  });
  document.querySelectorAll("[data-use]").forEach((b) => b.addEventListener("click", () => {
    const sug = state.assist[side].suggestions[+b.dataset.use];
    const cur = state.draft[side].trim();
    state.draft[side] = (cur ? cur + " " : "") + sug.argument;
    store(`draft-${s.id}-${side}`, state.draft[side]);
    renderHearing();
    const t = document.getElementById("draft"); if (t) { t.focus(); t.setSelectionRange(t.value.length, t.value.length); }
  }));
  const verdict = document.getElementById("verdict");
  if (verdict) verdict.addEventListener("click", async () => {
    state.busy = true; state.error = null; renderHearing();
    try { state.session = await api(`/api/sessions/${s.id}/verdict`, { method: "POST" }); }
    catch (e) { state.error = e.message; }
    state.busy = false; renderHearing(); schedulePoll();
  });
  document.querySelectorAll("[data-role]").forEach((b) => b.addEventListener("click", () => {
    state.role = b.dataset.role; store(`role-${s.id}`, state.role); renderHearing(); schedulePoll();
  }));
  const stream = document.getElementById("stream");
  if (stream) stream.scrollTop = stream.scrollHeight;
}

// ------------------------------------------------------------------ start
loadStatus();
setInterval(loadStatus, 30000);
route();
