// 메인 화면: 처방전 업로드 → 2초마다 상태 확인 → 조제지시서 승인 카드 (REQ-WEB-01·02·03·07)
(() => {
  const app = document.getElementById("app");
  if (!app) return;
  const CSRF = app.dataset.csrf;
  const LABELS = JSON.parse(app.dataset.labels);
  const TIMES = { morning: "아침", noon: "점심", evening: "저녁", bedtime: "취침 전" };
  const TIME_ORDER = ["morning", "noon", "evening", "bedtime"];
  const RISK = { none: "해당 없음", caution: "주의", warning: "경고", danger: "위험" };
  const POLL_MS = 2000;
  const LOW_CONF = 0.5;

  const card = document.getElementById("card");
  const cardEmpty = document.getElementById("card-empty");
  const list = document.getElementById("rx-list");
  let current = null;   // 지금 카드에 보이는 처방 ID
  let timer = null;

  // 작은 DOM 도우미: 글자는 항상 textContent로 넣는다 (OCR 글자에 HTML이 섞여도 안전)
  function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v == null) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      ...opts,
      headers: { "X-CSRF-Token": CSRF, ...(opts.headers || {}) },
      credentials: "same-origin",
    });
    let body = null;
    try { body = await res.json(); } catch { /* 본문 없음 */ }
    if (!res.ok) throw new Error((body && body.error) || `요청 실패 (${res.status})`);
    return body;
  }

  // ---------------------------------------------------------- 목록
  function upsertListItem(rx) {
    document.getElementById("rx-empty").hidden = true;
    let btn = list.querySelector(`[data-id="${rx.id}"]`);
    if (!btn) {
      btn = h("button", { type: "button", class: "rx-item", "data-id": rx.id },
        h("strong", {}, rx.display_no), h("span", { class: "badge" }), h("small", {}, "방금"));
      list.prepend(h("li", {}, btn));
    }
    const badge = btn.querySelector(".badge");
    badge.className = `badge ${rx.status}`;
    badge.textContent = LABELS[rx.status] || rx.status;
    for (const b of list.querySelectorAll(".rx-item")) b.toggleAttribute("aria-current", b.dataset.id === current);
  }

  list.addEventListener("click", (e) => {
    const btn = e.target.closest(".rx-item");
    if (btn) open(btn.dataset.id);
  });

  // ---------------------------------------------------------- 업로드
  const form = document.getElementById("upload-form");
  const uploadError = document.getElementById("upload-error");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    uploadError.hidden = true;
    const btn = document.getElementById("upload-btn");
    btn.disabled = true;
    btn.textContent = "올리는 중…";
    try {
      const rx = await api("/api/prescriptions", { method: "POST", body: new FormData(form) });
      form.reset();
      current = rx.id;
      upsertListItem(rx);
      open(rx.id);
    } catch (err) {
      uploadError.textContent = err.message;
      uploadError.hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "올리기";
    }
  });

  // ---------------------------------------------------------- 카드
  async function open(id) {
    clearTimeout(timer);
    current = id;
    history.replaceState(null, "", `#rx=${id}`);
    await refresh();
  }

  async function refresh() {
    const id = current;
    let rx;
    try {
      rx = await api(`/api/prescriptions/${id}`);
    } catch (err) {
      if (id !== current) return;
      render(h("p", { class: "form-error" }, err.message));
      return;
    }
    if (id !== current) return;   // 그사이 다른 처방을 골랐다
    upsertListItem(rx);
    render(view(rx));
    if (rx.status === "processing") {
      timer = setTimeout(refresh, POLL_MS);
    }
  }

  function render(node) {
    card.replaceChildren(node);
    card.hidden = false;
    cardEmpty.hidden = true;
  }

  function header(rx) {
    return h("header", { class: "rx-card-head" },
      h("h3", {}, `처방 ${rx.display_no}`),
      h("span", { class: `badge ${rx.status}` }, LABELS[rx.status] || rx.status));
  }

  function view(rx) {
    const wrap = h("div", {}, header(rx));
    if (rx.status === "failed") {
      wrap.append(h("p", { class: "form-error" }, rx.error_message),
        rx.error_code ? h("p", { class: "muted mono" }, rx.error_code) : null);
      return wrap;
    }
    if (rx.status === "waiting_tray" && rx.card && rx.card.order) {
      wrap.append(h("p", { class: "ok-note" }, "조제지시서를 승인했습니다. 조제 후 트레이 사진을 올려 주세요 (트레이 검수는 준비 중)."),
        itemsTable(rx.card.order, {}));
      return wrap;
    }
    if (rx.status === "processing") {
      wrap.append(h("p", { class: "processing" }, h("span", { class: "spinner", "aria-hidden": "true" }),
        "처방전을 읽고 조제지시서를 만드는 중입니다…"));
      return wrap;
    }
    if (rx.status === "waiting_approval" && rx.hitl_point === "rx_approval" && rx.card) {
      wrap.append(approvalCard(rx));
      return wrap;
    }
    wrap.append(h("p", { class: "muted" }, "이 단계의 카드는 아직 준비 중입니다."));
    return wrap;
  }

  function itemsTable(order, lines) {
    const rows = order.items.map((it) => {
      const confs = (it.source_line_ids || []).map((i) => lines[String(i)]).filter(Boolean).map((l) => l.conf);
      const low = confs.length && Math.min(...confs) < LOW_CONF;
      const times = TIME_ORDER.filter((t) => it.times.includes(t)).map((t) => TIMES[t]).join(" · ");
      const flagged = it.needs_pharmacist_check || !it.item_code;
      return h("tr", { class: flagged ? "flagged" : null },
        h("td", {}, h("div", {}, it.name),
          h("div", { class: "muted mono" }, it.item_code || "품목 없음"),
          ...(it.check_reasons || []).map((r) => h("div", { class: "reason" }, r))),
        h("td", { class: "num" }, it.count_per_dose),
        h("td", {}, times),
        h("td", { class: "num" }, `${it.days}일`),
        h("td", {}, low ? h("span", { class: "chip low" }, "인식 신뢰도 낮음") : h("span", { class: "muted" }, "-")));
    });
    return h("div", { class: "table-wrap" }, h("table", { class: "items" },
      h("thead", {}, h("tr", {}, h("th", {}, "약품"), h("th", {}, "1회"), h("th", {}, "복용 시점"),
        h("th", {}, "일수"), h("th", {}, "원문"))),
      h("tbody", {}, rows)));
  }

  function riskPanel(risk) {
    const level = risk.level || "none";
    const box = h("div", { class: `risk risk-${level}` },
      h("div", { class: "risk-head" }, h("strong", {}, `위험 신호: ${RISK[level]}`),
        risk.scope === "rx_only" ? h("span", { class: "muted" }, "처방약 기준 판단 (복용 약 미등록)") : null));
    if (risk.findings && risk.findings.length) {
      box.append(h("ul", {}, risk.findings.map((f) =>
        h("li", {}, h("span", { class: `chip ${f.level}` }, RISK[f.level] || f.level), " ", f.message))));
    }
    return box;
  }

  function approvalCard(rx) {
    const { order, ocr_lines: lines, blockers, needs_confirm: needsConfirm } = rx.card;
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const confirm = needsConfirm
      ? h("label", { class: "confirm" }, h("input", { type: "checkbox", id: "confirm-risk" }),
        "위험 신호와 확인 필요 항목을 확인했습니다")
      : null;
    const btn = h("button", { type: "button", disabled: blockers.length > 0 }, "조제지시서 승인");
    btn.addEventListener("click", async () => {
      err.hidden = true;
      btn.disabled = true;
      try {
        const confirmed = needsConfirm ? document.getElementById("confirm-risk").checked : false;
        await api(`/api/prescriptions/${rx.id}/approve`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirmed }),
        });
        await refresh();
      } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
        btn.disabled = false;
      }
    });
    return h("div", { class: "approval" },
      h("div", { class: "approval-grid" },
        rx.image_url ? h("figure", { class: "rx-image" },
          h("a", { href: rx.image_url, target: "_blank", rel: "noopener" },
            h("img", { src: rx.image_url, alt: `처방 ${rx.display_no} 원본 사진` })),
          h("figcaption", { class: "muted" }, "원본 (누르면 크게 보기)")) : null,
        h("div", { class: "approval-body" },
          h("p", { class: "muted" }, "원본과 대조한 뒤 승인하세요. 항목 수정은 다음 단계에서 붙습니다."),
          itemsTable(order, lines),
          riskPanel(order.risk_check || {}))),
      blockers.map((b) => h("p", { class: "form-error" }, b)),
      h("div", { class: "actions" }, confirm, btn),
      err);
  }

  // 새로고침해도 보던 처방을 다시 연다 (REQ-WEB-03)
  const m = location.hash.match(/^#rx=([0-9a-f-]{36})$/);
  if (m) open(m[1]);
})();
