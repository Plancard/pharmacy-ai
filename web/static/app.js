// 메인 화면: 처방전 업로드 → 2초마다 상태 확인 → 처방 타임라인(접수·승인·트레이 검수·완료) (REQ-WEB-01·02·03·07)
(() => {
  const app = document.getElementById("app");
  if (!app) return;
  const CSRF = app.dataset.csrf;
  const LABELS = JSON.parse(app.dataset.labels);
  const TZ = app.dataset.tz || "Asia/Seoul";
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

  function fmtTime(iso) {
    return iso ? new Date(iso).toLocaleString("ko-KR", {
      timeZone: TZ, month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) : "";
  }

  // 처방 타임라인 (REQ-WEB-07): 접수 → 조제지시서 승인 → 트레이 검수 → 완료가 위에서 아래로 쌓인다.
  // 색만으로 구분하지 않도록 단계마다 상태 글자(완료·진행 중·대기·실패·취소됨)를 함께 적는다.
  const STEP_STATE = { done: "완료", active: "진행 중", pending: "대기", error: "실패", cancelled: "취소됨" };

  function step(state, title, at, ...body) {
    return h("li", { class: `tl-item ${state}`, "aria-current": state === "active" ? "step" : null },
      h("span", { class: "tl-dot", "aria-hidden": "true" }),
      h("div", { class: "tl-main" },
        h("div", { class: "tl-head" }, h("strong", {}, title),
          at ? h("time", { class: "muted", datetime: at }, fmtTime(at)) : null,
          h("span", { class: "tl-state" }, STEP_STATE[state])),
        ...body.filter(Boolean)));
  }

  function spinner(text) {
    return h("p", { class: "processing" }, h("span", { class: "spinner", "aria-hidden": "true" }), text);
  }

  function failure(rx) {
    return [h("p", { class: "form-error" }, rx.error_message || "처리 중 오류가 났습니다."),
      rx.error_code ? h("p", { class: "muted mono" }, rx.error_code) : null];
  }

  function view(rx) {
    const events = rx.timeline || [];
    const received = events.find((e) => e.event === "received");
    const responses = events.filter((e) => e.event === "response" && e.hitl_point === "rx_approval");
    const approval = responses.filter((e) => e.action !== "edit").pop();
    const edits = responses.filter((e) => e.action === "edit");
    const order = rx.card && rx.card.order;
    const s = rx.status;
    const steps = [step("done", "처방전 접수", received && received.at,
      h("p", { class: "muted" }, "처방전 사진을 올렸습니다."))];

    // 조제지시서 (HITL ①). 약사가 고칠 때마다 '수정 → 다시 계산'이 한 단계씩 쌓인다
    edits.forEach((e, i) => steps.push(step("done", `조제지시서 수정 ${i + 1}회 (다시 계산)`, e.at)));
    if (approval) {
      steps.push(step("done", "조제지시서 승인", approval.at,
        order ? h("details", {}, h("summary", {}, "승인한 조제지시서 보기"), itemsTable(order, {})) : null));
    } else if (s === "waiting_approval" && rx.hitl_point === "rx_approval" && rx.card) {
      steps.push(step("active", "조제지시서 확인·승인", null, approvalCard(rx)));
    } else if (s === "failed") {
      steps.push(step("error", "조제지시서 작성", rx.updated_at, ...failure(rx)));
    } else if (s === "cancelled") {
      steps.push(step("cancelled", "조제지시서 확인·승인", rx.updated_at));
    } else {
      steps.push(step("active", "조제지시서 작성", null, spinner("처방전을 읽고 조제지시서와 조제 계획을 만드는 중입니다…")));
    }
    const flowOpen = !["failed", "cancelled"].includes(s);

    // 트레이 검수 (트레이 사진 업로드·검수 결과 카드는 11/18에 연결)
    if (!approval) {
      steps.push(step(flowOpen ? "pending" : (s === "failed" ? "pending" : "cancelled"), "트레이 검수", null,
        flowOpen ? h("p", { class: "muted" }, "조제지시서를 승인하면 조제 후 트레이 사진을 올릴 수 있습니다.") : null));
    } else if (s === "waiting_tray") {
      steps.push(step("active", "트레이 검수", null,
        h("p", { class: "ok-note" }, "조제지시서를 승인했습니다. 조제 후 트레이 사진을 올려 주세요 (트레이 검수는 준비 중)."),
        order ? h("details", {}, h("summary", {}, "승인한 조제지시서 보기"), itemsTable(order, {})) : null));
    } else if (s === "processing") {
      steps.push(step("active", "승인 내용 반영", null, spinner("승인한 내용으로 조제 계획을 확정하는 중입니다…")));
    } else if (s === "done") {
      steps.push(step("done", "트레이 검수", null));
    } else if (s === "failed") {
      steps.push(step("error", "승인 이후 처리", rx.updated_at, ...failure(rx)));
    } else if (s === "cancelled") {
      steps.push(step("cancelled", "트레이 검수", null));
    }

    // 마무리
    if (s === "done") steps.push(step("done", "조제 완료", rx.updated_at));
    else if (s === "cancelled") steps.push(step("cancelled", "취소됨", rx.updated_at));
    else if (s !== "failed") steps.push(step("pending", "조제 완료", null));

    return h("div", {}, header(rx), h("ol", { class: "timeline" }, steps));
  }

  // onSelect가 있으면 줄을 누르거나 Enter로 고를 수 있다 (F-WEB-06 원문 위치 표시)
  function itemsTable(order, lines, onSelect) {
    const rows = order.items.map((it, idx) => {
      const confs = (it.source_line_ids || []).map((i) => lines[String(i)]).filter(Boolean).map((l) => l.conf);
      const low = confs.length && Math.min(...confs) < LOW_CONF;
      const times = TIME_ORDER.filter((t) => it.times.includes(t)).map((t) => TIMES[t]).join(" · ");
      const flagged = it.needs_pharmacist_check || !it.item_code;
      const cls = [low ? "lowconf" : (flagged ? "flagged" : ""), onSelect ? "selectable" : ""].join(" ").trim();
      const tr = h("tr", { class: cls || null, tabindex: onSelect ? "0" : null,
        title: onSelect ? "누르면 원본에서 위치를 표시합니다" : null },
        h("td", {}, h("div", {}, it.name),
          h("div", { class: "muted mono" }, it.item_code || "품목 없음"),
          ...(it.check_reasons || []).map((r) => h("div", { class: "reason" }, r))),
        h("td", { class: "num" }, it.count_per_dose),
        h("td", {}, times),
        h("td", { class: "num" }, `${it.days}일`),
        h("td", {}, low ? h("span", { class: "chip low" }, "인식 신뢰도 낮음") : h("span", { class: "muted" }, "-")));
      if (onSelect) {
        tr.addEventListener("click", () => onSelect(idx, tr));
        tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(idx, tr); } });
      }
      return tr;
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

  // 조제 계획(A.4): 일수별로 시간대마다 넣을 약과 알 수. plan이 없으면 이유(notice)만 보여 준다.
  function planPanel(order, plan, notice) {
    if (!plan) {
      return h("div", { class: "risk", role: "status" }, h("strong", {}, "조제 계획"),
        h("p", { class: "form-error" }, notice || "조제 계획을 만들지 못했습니다."),
        h("p", { class: "muted" }, "항목을 수정해 다시 계산하기 전에는 승인할 수 없습니다."));
    }
    const names = Object.fromEntries(order.items.map((it) => [it.item_code, it.name]));
    const rows = plan.plan.map((c) => h("tr", {},
      h("td", {}, `${c.day}일차`), h("td", {}, TIMES[c.time] || c.time),
      h("td", {}, c.items.map((i) => `${names[i.item_code] || i.item_code} ${i.count}정`).join(", "))));
    return h("div", { class: "risk" }, h("strong", {}, `조제 계획 (${plan.days}일)`),
      h("table", { class: "items" }, h("thead", {}, h("tr", {}, h("th", {}, "일"), h("th", {}, "시간"), h("th", {}, "넣을 약"))),
        h("tbody", {}, rows)));
  }

  // 수정 폼 (HITL ① edit): 이름·1회 알 수·복용 시간·일수를 고치거나 항목을 지우고 새 항목을 더한다.
  // 보내면 서버가 매칭·대조·조제 계획을 다시 계산하고 승인 대기로 돌아온다. 여기서는 자체 오류만 보인다.
  function editForm(rx, onClose) {
    const order = rx.card.order;
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const body = h("tbody", {});
    function row(it, index) {
      const times = h("span", { class: "time-checks" }, TIME_ORDER.map((t) =>
        h("label", {}, h("input", { type: "checkbox", value: t, checked: !!(it && it.times.includes(t)) }), TIMES[t])));
      const tr = h("tr", { "data-index": index == null ? "" : String(index) },
        h("td", {}, h("input", { type: "text", class: "f-name", value: it ? it.name : "", maxlength: "100", "aria-label": "약품 이름" })),
        h("td", {}, h("input", { type: "number", class: "f-count", min: "1", max: "10", value: it ? it.count_per_dose : 1, "aria-label": "1회 알 수" })),
        h("td", {}, times),
        h("td", {}, h("input", { type: "number", class: "f-days", min: "1", max: "30", value: it ? it.days : 3, "aria-label": "복용 일수" })),
        h("td", {}, h("button", { type: "button", class: "link", onclick: () => tr.remove() }, "삭제")));
      return tr;
    }
    order.items.forEach((it, i) => body.append(row(it, i)));
    const add = h("button", { type: "button", class: "link", onclick: () => body.append(row(null, null)) }, "+ 항목 추가");
    const send = h("button", { type: "button" }, "수정 내용으로 다시 계산");
    send.addEventListener("click", async () => {
      err.hidden = true;
      const items = [...body.children].map((tr) => ({
        index: tr.dataset.index === "" ? null : Number(tr.dataset.index),
        name: tr.querySelector(".f-name").value.trim(),
        count_per_dose: Number(tr.querySelector(".f-count").value),
        times: [...tr.querySelectorAll(".time-checks input:checked")].map((c) => c.value),
        days: Number(tr.querySelector(".f-days").value),
      }));
      send.disabled = true;
      try {
        await api(`/api/prescriptions/${rx.id}/approve`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "edit", items }),
        });
        await refresh();
      } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
        send.disabled = false;
      }
    });
    const cancel = h("button", { type: "button", class: "link", onclick: onClose }, "수정 취소");
    return h("div", { class: "edit-form" },
      h("p", { class: "muted" }, "이름을 바꾸면 약품을 다시 찾고, 조제 계획과 위험 신호도 다시 계산합니다."),
      h("table", { class: "items" },
        h("thead", {}, h("tr", {}, ["약품", "1회 알 수", "복용 시간", "일수", ""].map((t) => h("th", {}, t)))), body),
      h("div", { class: "actions" }, add, cancel, send), err);
  }

  function approvalCard(rx) {
    const { order, plan, notice, ocr_lines: lines, ocr_size: size, blockers, needs_confirm: needsConfirm, is_danger: isDanger } = rx.card;
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const confirm = needsConfirm
      ? h("label", { class: "confirm" }, h("input", { type: "checkbox", id: "confirm-risk" }),
        "위험 신호와 확인 필요 항목을 확인했습니다")
      : null;
    // 위험(danger)은 그대로 승인할 수 없다: 사유를 적어 진행하거나 항목을 수정한다 (F-WEB-07)
    const reasonBox = isDanger
      ? h("textarea", { id: "manual-reason", rows: "2", maxlength: "500", class: "reason-input",
        placeholder: "위험 항목을 확인한 사유 (예: 처방 의사에게 확인함)", "aria-label": "진행 사유" })
      : null;
    const btn = h("button", { type: "button", disabled: blockers.length > 0 },
      isDanger ? "사유 입력 후 진행" : "조제지시서 승인");
    btn.addEventListener("click", async () => {
      err.hidden = true;
      btn.disabled = true;
      try {
        const confirmed = needsConfirm ? document.getElementById("confirm-risk").checked : false;
        const payload = isDanger
          ? { action: "manual", reason: reasonBox.value.trim() }
          : { confirmed };
        if (isDanger && !payload.reason) throw new Error("사유를 적어 주세요.");
        await api(`/api/prescriptions/${rx.id}/approve`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        await refresh();
      } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
        btn.disabled = false;
      }
    });
    // F-WEB-06 처방 결과: 왼쪽 원본, 오른쪽 추출 항목. 항목을 누르면 그 줄의 OCR 위치(bbox)를 원본 위에 표시
    const boxes = h("div", { class: "ocr-boxes", "aria-hidden": "true" });
    const canLocate = !!(size && size.width && size.height);
    function select(idx, tr) {
      for (const r of tr.parentNode.children) r.classList.toggle("selected", r === tr);
      boxes.replaceChildren();
      for (const id of order.items[idx].source_line_ids || []) {
        const l = lines[String(id)];
        if (!l || !l.bbox) continue;
        const [x1, y1, x2, y2] = l.bbox;
        const b = h("div", { class: `ocr-box${l.conf < LOW_CONF ? " low" : ""}` });
        b.style.left = `${(x1 / size.width) * 100}%`;
        b.style.top = `${(y1 / size.height) * 100}%`;
        b.style.width = `${((x2 - x1) / size.width) * 100}%`;
        b.style.height = `${((y2 - y1) / size.height) * 100}%`;
        boxes.append(b);
      }
      if (boxes.firstChild) boxes.firstChild.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    const form = h("div", {});   // 수정 폼 자리 (2초 갱신은 처리 중에만 돌아서 입력 중인 폼이 지워지지 않는다)
    return h("div", { class: "approval" },
      h("div", { class: "approval-grid" },
        rx.image_url ? h("figure", { class: "rx-image" },
          h("a", { href: rx.image_url, target: "_blank", rel: "noopener", class: "img-wrap" },
            h("img", { src: rx.image_url, alt: `처방 ${rx.display_no} 원본 사진` }), boxes),
          h("figcaption", { class: "muted" }, "원본 (누르면 크게 보기)")) : null,
        h("div", { class: "approval-body" },
          h("p", { class: "muted" }, canLocate
            ? "항목을 누르면 원본에서 읽은 위치를 표시합니다. 노란 줄은 인식 신뢰도가 낮거나 확인이 필요한 항목입니다."
            : "원본과 대조한 뒤 승인하세요."),
          itemsTable(order, lines, canLocate && rx.image_url ? select : null),
          planPanel(order, plan, notice),
          riskPanel(order.risk_check || {}))),
      blockers.map((b) => h("p", { class: "form-error" }, b)),
      reasonBox,
      h("div", { class: "actions" }, isDanger ? null : confirm, btn,
        h("button", { type: "button", class: "link", onclick: () => {
          const f = editForm(rx, () => { form.replaceChildren(); });
          form.replaceChildren(f);
          f.querySelector("input").focus();
        } }, "항목 수정")),
      err, form);
  }

  // 새로고침해도 보던 처방을 다시 연다 (REQ-WEB-03)
  // 주소에 처방이 없으면 가장 최근 처방을 연다
  const m = location.hash.match(/^#rx=([0-9a-f-]{36})$/);
  const latest = list.querySelector(".rx-item");
  if (m) open(m[1]);
  else if (latest) open(latest.dataset.id);
})();
