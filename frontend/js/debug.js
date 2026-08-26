/**
 * ClearClause Pipeline Debugger (debug.js)
 * Vanilla JavaScript — no frameworks, no build step.
 *
 * Development-only tool. Never linked from the normal app. Listens on the
 * same BroadcastChannel app.js publishes to (live, full detail while both
 * tabs are open) and falls back to the sanitised localStorage summary on
 * load (no source/claim/evidence text is ever persisted there — see
 * app.js's publishDebugPayload for what each channel carries and why).
 *
 * These names MUST match frontend/js/app.js.
 */

(function () {
  "use strict";

  const DEBUG_CHANNEL_NAME = "clearclause_pipeline_debug";
  const DEBUG_STORAGE_KEY = "clearclause_pipeline_debug_runs";

  const STAGE_DEFS = [
    { key: "span_generation", label: "Span Generation" },
    { key: "simplification", label: "Simplification" },
    { key: "claim_extraction", label: "Claim Extraction" },
    { key: "evidence_retrieval", label: "Evidence Retrieval" },
    { key: "nli_verification", label: "NLI Verification" },
    { key: "run_logging", label: "Run Logging" },
  ];

  const STATUS_STYLES = {
    success: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    partial: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    fallback: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    ambiguous: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    warning: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    skipped: "bg-slate-500/15 text-slate-400 border-slate-500/30",
    failed: "bg-rose-500/15 text-rose-400 border-rose-500/30",
  };

  function statusClass(status) {
    return STATUS_STYLES[status] || "bg-slate-500/15 text-slate-400 border-slate-500/30";
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str ?? ""));
    return div.innerHTML;
  }

  function fmtPct(value) {
    return (value === null || value === undefined) ? "—" : `${(value * 100).toFixed(1)}%`;
  }

  /* --- DOM references --------------------------------------------------- */
  const emptyState = document.getElementById("empty-state");
  const runContent = document.getElementById("run-content");
  const connectionIndicator = document.getElementById("connection-indicator");
  const clearBtn = document.getElementById("clear-debug-btn");

  const runOverallBadge = document.getElementById("run-overall-badge");
  const runIdEl = document.getElementById("run-id");
  const runTimestampEl = document.getElementById("run-timestamp");
  const runProvisionIdEl = document.getElementById("run-provision-id");
  const runSimplificationModelEl = document.getElementById("run-simplification-model");
  const runNliModelEl = document.getElementById("run-nli-model");
  const runErrorMessageEl = document.getElementById("run-error-message");
  const stageGrid = document.getElementById("stage-grid");
  const claimsTbody = document.getElementById("claims-tbody");

  /* --- Rendering --------------------------------------------------------- */
  function renderRun(payload) {
    if (!payload) return;

    emptyState.classList.add("hidden");
    runContent.classList.remove("hidden");

    const status = payload.pipeline_status || {};
    const overall = status.overall || "unknown";

    runOverallBadge.textContent = overall;
    runOverallBadge.className = `px-2 py-0.5 rounded text-xs font-semibold uppercase border ${statusClass(overall)}`;

    runIdEl.textContent = payload.run_id || "—";
    runTimestampEl.textContent = payload.timestamp || "—";
    runProvisionIdEl.textContent = payload.provision_id || "—";

    const provenance = payload.provenance || {};
    runSimplificationModelEl.textContent = provenance.simplification_model || "—";
    runNliModelEl.textContent = provenance.nli_model || "—";

    if (payload.error) {
      runErrorMessageEl.textContent = `Failure detail (not shown in the normal UI): ${payload.error}`;
      runErrorMessageEl.classList.remove("hidden");
    } else {
      runErrorMessageEl.classList.add("hidden");
    }

    // Stage grid. "Input Validation" is not a pipeline_status stage — it's
    // enforced by Pydantic before run_pipeline() ever executes — so it's
    // shown as a fixed informational card rather than reading a field
    // that doesn't exist.
    stageGrid.innerHTML = "";
    stageGrid.appendChild(buildStageCard(
      "Input Validation",
      { status: "success", reason: "Enforced at the request boundary (Pydantic), before pipeline execution." },
    ));
    STAGE_DEFS.forEach(({ key, label }) => {
      stageGrid.appendChild(buildStageCard(label, status[key]));
    });

    // Claims table.
    claimsTbody.innerHTML = "";
    const claims = Array.isArray(payload.claims) ? payload.claims : [];
    if (claims.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = `<td class="px-3 py-3 text-slate-500" colspan="9">No claims recorded for this run.</td>`;
      claimsTbody.appendChild(row);
    } else {
      claims.forEach((claim) => claimsTbody.appendChild(buildClaimRow(claim)));
    }
  }

  function buildStageCard(label, stage) {
    const card = document.createElement("div");
    card.className = "bg-slate-900 border border-slate-800 rounded-lg p-3";
    if (!stage) {
      card.innerHTML = `
        <p class="text-xs font-semibold text-slate-300 mb-1">${escapeHtml(label)}</p>
        <p class="text-xs text-slate-500">No data</p>
      `;
      return card;
    }
    const badge = `<span class="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase border ${statusClass(stage.status)}">${escapeHtml(stage.status)}</span>`;
    const details = [];
    if (stage.method) details.push(`method: ${stage.method}`);
    if (stage.attempts !== undefined && stage.attempts !== null) details.push(`attempts: ${stage.attempts}`);
    if (stage.reason) details.push(`reason: ${stage.reason}`);

    card.innerHTML = `
      <div class="flex items-center justify-between mb-1.5">
        <p class="text-xs font-semibold text-slate-300">${escapeHtml(label)}</p>
        ${badge}
      </div>
      ${details.length ? `<p class="text-[11px] text-slate-500 leading-relaxed">${escapeHtml(details.join(" · "))}</p>` : ""}
    `;
    return card;
  }

  function buildClaimRow(claim) {
    const row = document.createElement("tr");
    const probs = claim.nli_probabilities || {};
    const warnings = [...(claim.extraction_warnings || []), ...(claim.verification_warnings || [])];

    row.innerHTML = `
      <td class="px-3 py-2 text-slate-200 align-top">
        <div class="font-semibold">${escapeHtml(claim.claim_id || "—")}</div>
        ${claim.claim_text ? `<div class="text-slate-500 mt-0.5 max-w-xs">${escapeHtml(claim.claim_text)}</div>` : ""}
      </td>
      <td class="px-3 py-2 text-slate-300 align-top">${escapeHtml(claim.evidence_span_id ?? "—")}</td>
      <td class="px-3 py-2 text-slate-300 align-top">${escapeHtml(claim.evidence_method ?? "—")}</td>
      <td class="px-3 py-2 align-top">${claim.evidence_ambiguity ? "yes" : "no"}</td>
      <td class="px-3 py-2 text-slate-300 align-top">${escapeHtml(claim.verification_label ?? "—")}</td>
      <td class="px-3 py-2 text-slate-300 align-top">${fmtPct(claim.verification_confidence)}</td>
      <td class="px-3 py-2 text-slate-400 align-top">${fmtPct(probs.entailment)} / ${fmtPct(probs.contradiction)} / ${fmtPct(probs.neutral)}</td>
      <td class="px-3 py-2 text-slate-400 align-top">${escapeHtml(claim.verification_reason ?? "—")}</td>
      <td class="px-3 py-2 text-amber-400 align-top">${warnings.length ? escapeHtml(warnings.join(", ")) : "—"}</td>
    `;
    return row;
  }

  /* --- Data sources -------------------------------------------------------
   * Live: BroadcastChannel (full payload, while this tab and the main app
   * are both open). Fallback: the latest entry in the sanitised
   * localStorage history (available even if this tab is opened after the
   * fact, or in a browser without BroadcastChannel support).
   * --------------------------------------------------------------------- */
  function loadLatestFromStorage() {
    try {
      const raw = localStorage.getItem(DEBUG_STORAGE_KEY);
      const runs = raw ? JSON.parse(raw) : [];
      return Array.isArray(runs) && runs.length > 0 ? runs[0] : null;
    } catch (_) {
      return null;
    }
  }

  const latest = loadLatestFromStorage();
  if (latest) {
    renderRun(latest);
  }

  if (typeof BroadcastChannel !== "undefined") {
    const channel = new BroadcastChannel(DEBUG_CHANNEL_NAME);
    channel.onmessage = (event) => {
      connectionIndicator.textContent = "Live — updated just now";
      renderRun(event.data);
    };
    connectionIndicator.textContent = latest ? "Showing last saved run — waiting for a live update" : "Waiting for a run…";
  } else {
    connectionIndicator.textContent = "BroadcastChannel unsupported — showing last saved run only";
  }

  clearBtn.addEventListener("click", () => {
    try {
      localStorage.removeItem(DEBUG_STORAGE_KEY);
    } catch (_) { }
    emptyState.classList.remove("hidden");
    runContent.classList.add("hidden");
    connectionIndicator.textContent = "Waiting for a run…";
  });
})();
