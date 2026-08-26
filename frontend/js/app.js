/**
 * ClearClause — App Controller (app.js)
 * Vanilla JavaScript — no frameworks.
 * Backend: FastAPI at http://localhost:8000
 * =========================================================================*/

(function () {
  "use strict";

  /* -----------------------------------------------------------------------
   * 1. BASE PATH
   * --------------------------------------------------------------------- */
  const basePath = window.CLEARCLAUSE_BASE_PATH || ".";

  /* -----------------------------------------------------------------------
   * 2. BACKEND CONFIG
   * Change this if your FastAPI runs on a different port.
   * --------------------------------------------------------------------- */
  const API_BASE = "http://localhost:8000";

  /* -----------------------------------------------------------------------
   * 2b. DEBUG TELEMETRY  (always-on, background only — never shown here)
   *
   * Publishes a snapshot of every /pipeline call — success, partial, or a
   * structured failure — so a developer can open frontend/debug.html in a
   * second tab and see internal stage execution without cluttering this
   * page. Two channels, deliberately different in what they carry:
   *
   *   - BroadcastChannel: the FULL payload (including claim/evidence
   *     text), live, in-memory only. Only reaches a debug tab that
   *     happens to be open right now; never written to disk.
   *   - localStorage: a SANITISED summary (ids, statuses, scores,
   *     warnings, probabilities — no source/claim/evidence text), capped
   *     to the last DEBUG_MAX_STORED_RUNS runs, so opening the debug page
   *     later still shows something without persisting anything a user
   *     pasted or the model generated.
   *
   * The channel/storage key names here MUST match frontend/js/debug.js.
   * --------------------------------------------------------------------- */
  const DEBUG_CHANNEL_NAME = "clearclause_pipeline_debug";
  const DEBUG_STORAGE_KEY = "clearclause_pipeline_debug_runs";
  const DEBUG_MAX_STORED_RUNS = 5;

  const debugChannel = (typeof BroadcastChannel !== "undefined")
    ? new BroadcastChannel(DEBUG_CHANNEL_NAME)
    : null;

  function sanitiseClaimForStorage(claim) {
    return {
      claim_id: claim.claim_id,
      evidence_span_id: claim.evidence_span_id ?? null,
      evidence_method: claim.evidence_method ?? null,
      evidence_ambiguity: claim.evidence_ambiguity ?? null,
      verification_label: claim.verification_label ?? null,
      verification_confidence: claim.verification_confidence ?? null,
      nli_probabilities: claim.nli_probabilities ?? null,
      verification_reason: claim.verification_reason ?? null,
      verification_warnings: claim.verification_warnings ?? null,
      extraction_warnings: claim.extraction_warnings ?? null,
    };
  }

  function publishDebugPayload(fullPayload) {
    if (debugChannel) {
      try {
        debugChannel.postMessage(fullPayload);
      } catch (_) {
        // Structured-clone failure on an unusual payload shape — never
        // let a debug-telemetry issue affect the actual app.
      }
    }

    try {
      const summary = {
        run_id: fullPayload.run_id ?? null,
        timestamp: fullPayload.timestamp,
        provision_id: fullPayload.provision_id ?? null,
        pipeline_status: fullPayload.pipeline_status ?? null,
        provenance: fullPayload.provenance ?? null,
        error: fullPayload.error ?? null,
        claims: Array.isArray(fullPayload.claims)
          ? fullPayload.claims.map(sanitiseClaimForStorage)
          : [],
      };
      const existingRaw = localStorage.getItem(DEBUG_STORAGE_KEY);
      const existing = existingRaw ? JSON.parse(existingRaw) : [];
      const updated = [summary, ...existing].slice(0, DEBUG_MAX_STORED_RUNS);
      localStorage.setItem(DEBUG_STORAGE_KEY, JSON.stringify(updated));
    } catch (_) {
      // localStorage unavailable/full/private-mode — non-critical, the
      // live BroadcastChannel path above still worked if a tab was open.
    }
  }

  /* -----------------------------------------------------------------------
   * 3. COMPONENT LOADING  (local static HTML snippets only)
   * --------------------------------------------------------------------- */
  async function loadComponent(containerId, componentPath) {
    const container = document.getElementById(containerId);
    if (!container) return;
    try {
      const res = await fetch(`${basePath}/${componentPath}`);
      if (!res.ok) throw new Error(`Failed to load ${componentPath}`);
      container.innerHTML = await res.text();
    } catch (err) {
      console.error(err);
      container.innerHTML = `<p class="text-red-500 text-xs p-2">Component load error.</p>`;
    }
  }

  /* -----------------------------------------------------------------------
   * 4. LINK ADJUSTMENT
   * Rewrite navbar hrefs so links resolve correctly from both index.html
   * (base = ".") and pages/simplify.html (base = "..").
   * --------------------------------------------------------------------- */
  function adjustNavLinks() {
    document.querySelectorAll('[data-nav="home"]').forEach((a) => {
      a.setAttribute("href", `${basePath}/index.html`);
    });
    document.querySelectorAll('[data-nav="simplify"]').forEach((a) => {
      a.setAttribute("href", `${basePath}/pages/simplify.html`);
    });
    document.querySelectorAll('img[data-asset]').forEach((img) => {
      const assetPath = img.getAttribute('data-asset');
      img.setAttribute("src", `${basePath}/${assetPath}`);
    });
  }

  /* -----------------------------------------------------------------------
   * 5. BACKEND FUNCTIONS
   * --------------------------------------------------------------------- */

  /**
   * BACKEND CALL: simplification + span splitting.
   *
   * Calls POST http://localhost:8000/pipeline
   * Request:  { "text": "<user pasted statute provision>" }
   * Response: { "source_spans", "simplified_text", "claims" }
   *
   * Returns the subset needed for the simplification step:
   *   { source_text, source_spans, simplified_text, claims }
   *
   * Note: Your FastAPI /pipeline endpoint returns everything in one call,
   * so we capture claims here too and carry them through to runClaimExtraction.
   *
   * @param {string} inputText  The raw statute text from the textarea.
   * @returns {Promise<Object>}
   */
  async function runSimplification(inputText) {
    const res = await fetch(`${API_BASE}/pipeline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText }),
    });

    if (!res.ok) {
      // `detail` can be one of three shapes depending on where the
      // request failed:
      //   - a structured object {message, spans, pipeline_status} from a
      //     core-stage failure (span generation / simplification) — the
      //     technical message is kept on the thrown Error for the debug
      //     publish step, but never shown to the user directly.
      //   - a Pydantic validation-error list (request rejected before the
      //     pipeline ever ran, e.g. empty/oversized input).
      //   - a plain string, for anything else.
      let userMessage = `Server error ${res.status}`;
      let debugDetail = null;
      try {
        const errJson = await res.json();
        const detail = errJson.detail;
        if (typeof detail === "string") {
          userMessage = detail;
        } else if (detail && typeof detail === "object" && !Array.isArray(detail)) {
          userMessage = detail.message || userMessage;
          debugDetail = detail;
        } else if (Array.isArray(detail) && detail.length > 0 && detail[0].msg) {
          userMessage = detail[0].msg;
        }
      } catch (_) { }
      const error = new Error(userMessage);
      error.debugPayload = debugDetail;
      error.httpStatus = res.status;
      throw error;
    }

    const data = await res.json();

    // Your backend /pipeline returns:
    //   { provision_id, simplified_text, claims, spans, run_id,
    //     pipeline_status, provenance }
    // We map this to the shape the UI expects, and carry the new
    // diagnostic fields through under _-prefixed keys purely for the
    // debug-telemetry publish step — the normal UI never reads them.
    return {
      source_text: inputText,           // backend echoes text via provision_id; we keep original
      source_spans: data.spans,         // backend field is "spans"
      simplified_text: data.simplified_text,
      _claims: data.claims,             // carry claims through to runClaimExtraction
      _run_id: data.run_id,
      _provision_id: data.provision_id,
      _pipeline_status: data.pipeline_status,
      _provenance: data.provenance,
    };
  }

  /**
   * BACKEND CALL: claim extraction (already done inside /pipeline).
   *
   * Because your FastAPI /pipeline endpoint returns claims alongside spans
   * and simplified text in one response, this function simply unpacks the
   * claims that runSimplification already fetched.
   *
   * If you later split this into a separate /claims endpoint, replace the
   * body of this function with a real fetch call to that endpoint.
   *
   * @param {Object} simplificationResult  Output of runSimplification().
   * @returns {Promise<Object>}  { claims }
   */
  async function runClaimExtraction(simplificationResult) {
    // Claims were already returned by /pipeline — no second call needed.
    return {
      claims: simplificationResult._claims,
    };

    // --- Future separate /claims endpoint (if you split the pipeline later) ---
    // const res = await fetch(`${API_BASE}/claims`, {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify({
    //     source_text: simplificationResult.source_text,
    //     source_spans: simplificationResult.source_spans,
    //     simplified_text: simplificationResult.simplified_text,
    //   }),
    // });
    // if (!res.ok) throw new Error(`Claims error ${res.status}`);
    // return await res.json();
  }

  /* -----------------------------------------------------------------------
   * 6. UTILITY
   * --------------------------------------------------------------------- */
  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str ?? ""));
    return div.innerHTML;
  }

  /* -----------------------------------------------------------------------
   * 7. SIMPLIFY PAGE LOGIC  (only runs if #simplify-form exists)
   * --------------------------------------------------------------------- */
  function initSimplifyPage() {

    /* --- DOM References ------------------------------------------------- */
    const form = document.getElementById("simplify-form");
    if (!form) return; // Not on the simplify page — bail out safely.

    const textarea = document.getElementById("statute-input");
    const inputPanel = document.getElementById("input-panel");
    const loadingPanel = document.getElementById("loading-panel");
    const loadingText = document.getElementById("loading-text");
    const resultsPanel = document.getElementById("results-panel");
    const validationMsg = document.getElementById("validation-msg");
    const statusLive = document.getElementById("status-live");

    /* --- Input validation ----------------------------------------------- */
    function showValidationError(msg) {
      validationMsg.textContent = msg;
      validationMsg.classList.remove("hidden");
      textarea.setAttribute("aria-invalid", "true");
    }

    function clearValidationError() {
      validationMsg.textContent = "";
      validationMsg.classList.add("hidden");
      textarea.removeAttribute("aria-invalid");
    }

    /* --- Loading state -------------------------------------------------- */
    function showLoading(msg) {
      loadingPanel.classList.remove("hidden");
      loadingText.textContent = msg;
      statusLive.textContent = msg;
    }

    function hideLoading() {
      loadingPanel.classList.add("hidden");
      loadingText.textContent = "";
      statusLive.textContent = "";
    }

    /* --- Error banner --------------------------------------------------- */
    // Shows a dismissible red error card above the input panel when the
    // backend is unreachable or returns an error.
    function showErrorBanner(msg) {
      // Remove any existing banner first
      const existing = document.getElementById("error-banner");
      if (existing) existing.remove();

      const banner = document.createElement("div");
      banner.id = "error-banner";
      banner.className = "mb-4 flex items-start gap-3 bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700 fade-in";
      banner.innerHTML = `
        <svg class="w-4 h-4 mt-0.5 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"/>
        </svg>
        <div>
          <p class="font-medium">Backend error</p>
          <p class="mt-0.5 text-red-600">${escapeHtml(msg)}</p>
          <p class="mt-1 text-xs text-red-500">Make sure your FastAPI server is running: <code class="bg-red-100 px-1 rounded">uvicorn backend.main:app --reload</code></p>
        </div>
        <button onclick="this.parentElement.remove()" class="ml-auto text-red-400 hover:text-red-600 flex-shrink-0" aria-label="Dismiss">
          <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12"/></svg>
        </button>
      `;

      // Insert before the input panel
      inputPanel.parentNode.insertBefore(banner, inputPanel);
    }

    function clearErrorBanner() {
      const existing = document.getElementById("error-banner");
      if (existing) existing.remove();
    }

    /* --- Render functions ----------------------------------------------- */
    function renderResults(finalResult) {
      resultsPanel.innerHTML = "";

      // Section 0 — Original Text
      const originalSection = document.createElement("section");
      originalSection.className = "card-enter mb-8";
      originalSection.innerHTML = `
        <h2 class="text-lg font-semibold text-slate-800 mb-3">Original Text</h2>
        <div class="bg-slate-50 rounded-lg border border-slate-200 p-5">
          <p class="text-slate-600 leading-relaxed italic">"${escapeHtml(finalResult.source_text)}"</p>
        </div>
      `;
      resultsPanel.appendChild(originalSection);

      // Section A — Simplified Text
      // The simplified text is split on blank lines and each paragraph is
      // rendered as its own <p>. This keeps paragraph structure (the prompt
      // asks for a blank line between separate rules or punishment branches)
      // without using pre-wrap, so stray double-spaces and single soft line
      // breaks are still collapsed by normal HTML wrapping. Each paragraph is
      // escaped individually.
      const simplifiedParagraphs = String(finalResult.simplified_text ?? "")
        .split(/\n\s*\n/)
        .map((para) => para.trim())
        .filter((para) => para.length > 0)
        .map(
          (para) =>
            `<p class="text-slate-700 leading-relaxed mb-3 last:mb-0">${escapeHtml(para)}</p>`
        )
        .join("");

      const simplifiedSection = document.createElement("section");
      simplifiedSection.className = "card-enter";
      simplifiedSection.innerHTML = `
        <h2 class="text-lg font-semibold text-slate-800 mb-3">Simplified Text</h2>
        <div class="bg-white rounded-lg border border-slate-200 accent-card-blue p-5">
          ${simplifiedParagraphs}
        </div>
      `;
      resultsPanel.appendChild(simplifiedSection);

      // Section B — Atomic Claims
      const claimsSection = document.createElement("section");
      claimsSection.className = "card-enter mt-8";
      let claimsHTML = `<h2 class="text-lg font-semibold text-slate-800 mb-3">Atomic Claims</h2>`;
      claimsHTML += `<p class="text-xs text-slate-500 mb-3 italic">This percentage represents the model's confidence for this individual claim-evidence pair. It is not the model's overall accuracy or a guarantee of legal correctness.</p>`;
      finalResult.claims.forEach((claim, idx) => {
        // Evidence label. When the backend joined a headless continuation
        // branch to its preceding span, evidence_span_ids holds BOTH ids and
        // they are shown as "P1 + P2". The evidence text rendered below is the
        // same combined text the NLI model received, so what the user sees is
        // what the model scored.
        const evidenceIds = Array.isArray(claim.evidence_span_ids)
          ? claim.evidence_span_ids.filter(Boolean)
          : [];
        const spanLabel =
          evidenceIds.length > 1
            ? evidenceIds.join(" + ")
            : claim.evidence_span_id ?? "—";
        const isCombinedEvidence = evidenceIds.length > 1;

        let badgeColorClass = "bg-slate-100 text-slate-800";
        const labelVal = claim.verification_label ?? "unverified";
        if (labelVal === "supported") {
          badgeColorClass = "bg-green-100 text-green-800";
        } else if (labelVal === "unsupported") {
          badgeColorClass = "bg-red-100 text-red-800";
        } else if (labelVal === "uncertain") {
          badgeColorClass = "bg-amber-100 text-amber-800";
        }

        // Internal reason codes (verification_reason) never surface here —
        // the user sees a plain "Unable to verify" regardless of whether
        // that was no evidence, an NLI model/inference failure, or an
        // oversized input; the real reason stays in the debug page/API.
        const displayLabel = labelVal === "unverified" ? "Unable to verify" : labelVal;
        const labelBadge = `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold ${badgeColorClass}">${escapeHtml(displayLabel)}</span>`;

        // NLI label confidence line — hidden entirely when the backend didn't
        // run verification (no evidence linked, or verification failed).
        // This is the softmax probability of the predicted NLI class, not a
        // calibrated probability that the label is correct.
        const hasConfidence = claim.verification_confidence !== null && claim.verification_confidence !== undefined;
        const confidenceLine = hasConfidence
          ? `<p class="text-xs text-slate-500 mt-2">NLI label confidence: ${(claim.verification_confidence * 100).toFixed(2)}%</p>`
          : "";

        // Evidence actually sent to the NLI model. Always rendered when
        // present, and explicitly flagged when it is a combined
        // previous + continuation-branch premise, so the backend can never
        // silently verify against text the user was not shown.
        const evidenceBlock = claim.evidence_text
          ? `<details class="mt-2 text-xs text-slate-500"${isCombinedEvidence ? " open" : ""}>
              <summary class="cursor-pointer select-none hover:text-slate-700">${
                isCombinedEvidence
                  ? `Evidence sent to model (combined ${escapeHtml(spanLabel)})`
                  : "Evidence sent to model"
              }</summary>
              ${
                isCombinedEvidence
                  ? `<p class="mt-1 text-slate-400 italic">This branch is headless on its own, so the preceding span was joined to it before verification.</p>`
                  : ""
              }
              <p class="mono-span mt-1 text-slate-600 bg-slate-50 rounded p-2">${escapeHtml(claim.evidence_text)}</p>
            </details>`
          : "";

        // "Model details" is collapsed by default and only rendered when
        // nli_probabilities is present, so older/incomplete responses
        // (missing the field entirely) don't break rendering.
        const probs = claim.nli_probabilities;
        const modelDetails = probs
          ? `<details class="mt-2 text-xs text-slate-500">
              <summary class="cursor-pointer select-none hover:text-slate-700">Model details</summary>
              <ul class="mt-1 ml-4 list-disc">
                <li>Entailment: ${((probs.entailment ?? 0) * 100).toFixed(2)}%</li>
                <li>Contradiction: ${((probs.contradiction ?? 0) * 100).toFixed(2)}%</li>
                <li>Neutral: ${((probs.neutral ?? 0) * 100).toFixed(2)}%</li>
              </ul>
            </details>`
          : "";

        claimsHTML += `
          <div class="bg-white rounded-lg border border-slate-200 p-4 mb-3 fade-in-up" data-delay="${idx + 1}">
            <div class="flex flex-wrap items-start gap-2 mb-1.5">
              <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-100 text-emerald-800">${escapeHtml(claim.claim_id)}</span>
              <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-purple-100 text-purple-800">${escapeHtml(spanLabel)}</span>
              ${labelBadge}
            </div>
            <p class="text-slate-700 text-sm leading-relaxed">${escapeHtml(claim.claim_text)}</p>
            ${confidenceLine}
            ${evidenceBlock}
            ${modelDetails}
          </div>
        `;
      });

      const hasVerification = finalResult.claims.some(c => c.verification_label !== undefined && c.verification_label !== null);
      if (!hasVerification) {
        claimsHTML += `<p class="text-xs text-slate-400 mt-2 italic">NLI verification coming soon</p>`;
      }

      claimsSection.innerHTML = claimsHTML;
      resultsPanel.appendChild(claimsSection);


      // Section C — Source Spans
      const spansSection = document.createElement("section");
      spansSection.className = "card-enter mt-8";
      let spansHTML = `<h2 class="text-lg font-semibold text-slate-800 mb-3">Source Spans</h2>`;
      finalResult.source_spans.forEach((span, idx) => {
        spansHTML += `
          <div class="bg-white rounded-lg border border-slate-200 p-4 mb-3 fade-in-up" data-delay="${idx + 1}">
            <div class="flex flex-wrap items-center gap-2 mb-1.5">
              <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-purple-100 text-purple-800">${escapeHtml(span.span_id)}</span>
              <span class="text-xs text-slate-400">chars ${span.start}–${span.end}</span>
            </div>
            <p class="mono-span text-slate-600 bg-slate-50 rounded p-3">${escapeHtml(span.text)}</p>
          </div>
        `;
      });
      spansSection.innerHTML = spansHTML;
      resultsPanel.appendChild(spansSection);

      // Reset button
      const resetWrap = document.createElement("div");
      resetWrap.className = "mt-8 text-center card-enter";
      resetWrap.innerHTML = `
        <button id="reset-btn"
                class="btn-lift inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-slate-300 text-slate-600 text-sm font-medium bg-white hover:bg-slate-50 transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
          <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" /></svg>
          Clear / Try another
        </button>
      `;
      resultsPanel.appendChild(resetWrap);

      document.getElementById("reset-btn").addEventListener("click", resetPage);

      resultsPanel.classList.remove("hidden");
      statusLive.textContent = "Results are ready.";
    }

    /* --- Reset logic ---------------------------------------------------- */
    function resetPage() {
      textarea.value = "";
      clearValidationError();
      clearErrorBanner();
      hideLoading();
      resultsPanel.innerHTML = "";
      resultsPanel.classList.add("hidden");
      inputPanel.classList.remove("hidden");
      statusLive.textContent = "";
      textarea.focus();
    }

    /* --- Event listeners ------------------------------------------------ */
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      clearValidationError();
      clearErrorBanner();

      const inputText = textarea.value.trim();

      // Validate
      if (!inputText) {
        showValidationError("Please paste a statute provision before simplifying.");
        textarea.focus();
        return;
      }

      // Hide input, show loading
      inputPanel.classList.add("hidden");

      // Step 1 — Simplification + span splitting (single /pipeline call)
      showLoading("Simplifying provision…");
      let simplificationResult;
      try {
        simplificationResult = await runSimplification(inputText);
      } catch (err) {
        console.error(err);
        hideLoading();
        inputPanel.classList.remove("hidden");
        // The real technical reason (err.message / err.debugPayload) is
        // never shown here — only in the debug page / API / logs. The
        // normal UI always shows the same plain outcome.
        publishDebugPayload({
          run_id: err.debugPayload ? err.debugPayload.run_id : null,
          timestamp: new Date().toISOString(),
          provision_id: err.debugPayload ? err.debugPayload.provision_id : null,
          pipeline_status: err.debugPayload ? err.debugPayload.pipeline_status : null,
          spans: err.debugPayload ? err.debugPayload.spans : [],
          claims: [],
          provenance: null,
          error: err.message,
        });
        showErrorBanner("The simplification could not be completed. Please try again.");
        return;
      }

      // Step 2 — Claim extraction (unpacks from pipeline response)
      showLoading("Extracting atomic claims…");
      let claimsResult;
      try {
        claimsResult = await runClaimExtraction(simplificationResult);
      } catch (err) {
        console.error(err);
        hideLoading();
        inputPanel.classList.remove("hidden");
        showErrorBanner("The simplification could not be completed. Please try again.");
        return;
      }

      // Combine into finalResult
      const finalResult = {
        source_text: simplificationResult.source_text,
        source_spans: simplificationResult.source_spans,
        simplified_text: simplificationResult.simplified_text,
        claims: claimsResult.claims,
      };

      publishDebugPayload({
        run_id: simplificationResult._run_id,
        timestamp: new Date().toISOString(),
        provision_id: simplificationResult._provision_id,
        pipeline_status: simplificationResult._pipeline_status,
        provenance: simplificationResult._provenance,
        spans: simplificationResult.source_spans,
        claims: claimsResult.claims,
      });

      // Render
      hideLoading();
      renderResults(finalResult);
    });

    // Clear validation on typing
    textarea.addEventListener("input", () => {
      if (textarea.value.trim()) clearValidationError();
    });
  }

  /* -----------------------------------------------------------------------
   * 8. INIT
   * --------------------------------------------------------------------- */
  async function init() {
    await Promise.all([
      loadComponent("navbar-container", "components/navbar.html"),
      loadComponent("disclaimer-container", "components/disclaimer.html"),
    ]);

    adjustNavLinks();
    initSimplifyPage();
    document.body.classList.add("fade-in");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
