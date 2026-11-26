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
  const CELL = { ok: "일치", mismatch: "불일치", need_review: "확인 필요" };
  const CELL_REASON = { boundary: "칸 경계에 걸침", overlap: "알약 겹침", rotated: "트레이가 뒤집혀 찍힘" };
  const DAYS = [1, 2, 3];   // 트레이 3열 (1~3일), 4행은 TIME_ORDER
  const POLL_MS = 2000;
  const LOW_CONF = 0.5;

  const card = document.getElementById("card");
  const intake = document.getElementById("intake");   // 새 처방 접수 (처방을 고르지 않았을 때만 보인다)
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

  // 아이콘 (11/26 화면 디자인): base.html의 아이콘 묶음(_icons.html)을 <use>로 쓴다
  const SVG_NS = "http://www.w3.org/2000/svg";
  function icon(name) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }
  function setLabel(btn, name, text) { btn.replaceChildren(...(name ? [icon(name)] : []), text); }

  // 사진 아래 '원본 사진' 버튼 (새 창으로 연다). 설명 글 없이 작은 칩으로만 둔다 (11/26)
  function originalLink(url, label = "원본 사진") {
    return h("a", { href: url, target: "_blank", rel: "noopener", class: "chip-link" }, icon("image"), label);
  }
  // 사진 위 오른쪽 아래 돋보기 (누르면 크게 보인다는 표시 — 글 대신)
  function zoomHint() { return h("span", { class: "zoom-hint", "aria-hidden": "true" }, icon("zoom")); }

  // 사진 고르기 칸: 눌러서 고르거나 끌어다 놓는다. 고른 사진은 이름 · 크기 · 미리보기로 보인다.
  // input은 화면에서만 숨긴다 (display:none이면 required 안내가 뜨지 않는다)
  function fileDrop(input, hint = "눌러서 고르거나 사진을 끌어다 놓으세요 · JPG · PNG, 10MB 이하") {
    input.classList.add("drop-input");
    const name = h("span", { class: "drop-name" });
    const thumb = h("img", { class: "drop-thumb", alt: "", hidden: true });
    const zone = h("label", { class: "drop" }, input, icon("camera"),
      h("span", { class: "drop-text" }, h("strong", {}, "사진 선택"), name), thumb);
    function update() {
      const f = input.files && input.files[0];
      if (thumb.getAttribute("src")) URL.revokeObjectURL(thumb.src);
      if (f) {
        name.textContent = `${f.name} · ${Math.max(1, Math.round(f.size / 1024))}KB`;
        thumb.src = URL.createObjectURL(f);
        thumb.hidden = false;
      } else {
        name.textContent = hint;
        thumb.removeAttribute("src");
        thumb.hidden = true;
      }
      zone.classList.toggle("has-file", Boolean(f));
    }
    input.addEventListener("change", update);
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("over");
      if (e.dataTransfer && e.dataTransfer.files.length) { input.files = e.dataTransfer.files; update(); }
    });
    document.addEventListener("reset", (e) => { if (e.target === input.form) setTimeout(update); });
    update();
    return zone;
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
    let btn = document.querySelector(`.rx-item[data-id="${rx.id}"]`);
    if (!btn) {
      btn = h("button", { type: "button", class: "rx-item", "data-id": rx.id },
        h("strong", {}, rx.display_no), h("span", { class: "badge" }), h("small", {}, "방금"));
      list.prepend(h("li", {}, btn));
    }
    const badge = btn.querySelector(".badge");
    badge.className = `badge ${rx.status}`;
    badge.textContent = LABELS[rx.status] || rx.status;
    // 완료 · 취소가 되면 아래 '최근 완료 · 취소'로 옮긴다 (진행 중 목록에는 진행 중인 것만)
    const doneList = document.getElementById("rx-done");
    if (["done", "cancelled"].includes(rx.status) && btn.closest("#rx-list")) {
      doneList.prepend(btn.parentElement);
      document.getElementById("rx-done-head").hidden = false;
    }
    document.getElementById("rx-empty").hidden = list.children.length > 0;
    markCurrent();
  }

  function markCurrent() {
    for (const b of document.querySelectorAll(".rx-item")) b.toggleAttribute("aria-current", b.dataset.id === current);
  }

  // 새 처방 접수 화면: 고른 처방을 풀고 접수 칸만 보인다. 주소의 #rx=도 지운다 (새로고침하면 다시 접수 화면)
  function showIntake() {
    clearTimeout(timer);
    current = null;
    history.replaceState(null, "", location.pathname + location.search);
    card.replaceChildren();
    card.hidden = true;
    intake.hidden = false;
    markCurrent();
    document.getElementById("rx-image").focus();
  }

  document.getElementById("new-rx").addEventListener("click", showIntake);

  list.closest("aside").addEventListener("click", (e) => {
    const btn = e.target.closest(".rx-item");
    if (btn) open(btn.dataset.id);
  });

  // ---------------------------------------------------------- 업로드
  const form = document.getElementById("upload-form");
  const uploadError = document.getElementById("upload-error");
  const rxInput = document.getElementById("rx-image");
  const rxRow = rxInput.parentNode;                  // fileDrop이 input을 옮기기 전에 자리를 잡아 둔다
  rxRow.insertBefore(fileDrop(rxInput), rxRow.firstChild);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    uploadError.hidden = true;
    const btn = document.getElementById("upload-btn");
    btn.disabled = true;
    setLabel(btn, null, "올리는 중…");
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
      setLabel(btn, "upload", "올리기");
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
    intake.hidden = true;
  }

  // 처방 취소 (F-5, REQ-WEB-07): 완료 전이면 언제든. 승인된 건(트레이 대기)은 한 번 더 확인한다. 실패 카드에서는 [삭제].
  const CANCELLABLE = ["processing", "waiting_approval", "waiting_tray", "failed"];
  function cancelButton(rx) {
    if (!CANCELLABLE.includes(rx.status)) return null;
    const isFailed = rx.status === "failed";
    const btn = h("button", { type: "button", class: "danger-outline" }, icon(isFailed ? "trash" : "x"), isFailed ? "삭제" : "처방 취소");
    btn.addEventListener("click", async () => {
      const ask = isFailed
        ? `처방 ${rx.display_no}을(를) 삭제할까요? (취소됨으로 남습니다)`
        : `처방 ${rx.display_no}을(를) 취소할까요?`;
      if (!window.confirm(ask)) return;
      if (rx.status === "waiting_tray" && !window.confirm("이미 승인된 처방입니다. 정말 취소할까요?")) return;
      btn.disabled = true;
      try {
        await api(`/api/prescriptions/${rx.id}/cancel`, { method: "POST" });
      } catch (e) {
        window.alert(e.message);
      }
      await refresh();
    });
    return btn;
  }

  function header(rx) {
    return h("header", { class: "rx-card-head" },
      h("h3", {}, `처방 ${rx.display_no}`),
      h("span", { class: `badge ${rx.status}` }, LABELS[rx.status] || rx.status),
      cancelButton(rx));
  }

  // 진행 단계 표시 (11/26): 처방 카드 맨 위에 네 단계를 가로로 보인다. 아래 타임라인의 요약이라 같은 상태에서 정한다.
  // stage = 지금 머문 단계 번호(0~3), state = 그 단계의 상태(active · error · cancelled). 완료면 stage 4.
  const STAGES = [["scan", "처방전 인식"], ["approve", "조제지시서 승인"], ["tray", "트레이 검수"], ["done", "조제 완료"]];
  function progress(stage, state) {
    return h("ol", { class: "stepper", "aria-label": "진행 단계" }, STAGES.map(([ic, label], i) => {
      const st = i < stage ? "done" : i === stage ? state : "pending";
      return h("li", { class: `stepper-item ${st}`, "aria-current": i === stage && st === "active" ? "step" : null },
        h("span", { class: "stepper-dot" }, icon(st === "done" ? "check" : st === "error" ? "x" : ic)),
        h("span", { class: "stepper-label" }, label,
          h("small", {}, { done: "완료", active: "진행 중", pending: "대기", error: "실패", cancelled: "취소됨" }[st])));
    }));
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

  // Q6: 구조화에 실패한 줄을 펴진 처방전 위에 빨갛게 표시한다 (OCR 좌표는 펴진 이미지 기준이라 ?view=rect).
  function failedImage(rx) {
    const f = rx.failed_lines;
    const size = f && f.ocr_size;
    if (!rx.image_url || !size || !size.width || !size.height || !(f.lines || []).length) return null;
    const boxes = h("div", { class: "ocr-boxes", "aria-hidden": "true" });
    for (const l of f.lines) {
      const [x1, y1, x2, y2] = l.bbox;
      const b = h("div", { class: "ocr-box fail" });
      b.style.left = `${(x1 / size.width) * 100}%`;
      b.style.top = `${(y1 / size.height) * 100}%`;
      b.style.width = `${((x2 - x1) / size.width) * 100}%`;
      b.style.height = `${((y2 - y1) / size.height) * 100}%`;
      boxes.append(b);
    }
    return h("figure", { class: "rx-image failed-image" },
      h("div", { class: "img-wrap" },
        h("img", { src: `${rx.image_url}?view=rect`, alt: `처방 ${rx.display_no} 펴진 처방전 사진, 처리하지 못한 줄에 빨간 박스` }), boxes),
      h("figcaption", { class: "img-actions" }, h("span", { class: "muted" }, "빨간 박스: AI가 처리하지 못한 줄"),
        originalLink(rx.image_url)));
  }

  // Q6 [다시 시도]: 일시 오류(E_LLM·E_EXT_API·E_INTERRUPTED)로 실패했을 때만. 같은 처방전 사진으로 새 처방을 만들어 처음부터 다시 돌린다.
  function retryButton(rx) {
    if (!rx.retryable) return null;
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const btn = h("button", { type: "button" }, icon("retry"), "다시 시도");
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      err.hidden = true;
      try {
        const next = await api(`/api/prescriptions/${rx.id}/retry`, { method: "POST" });
        upsertListItem(next);
        await open(next.id);
      } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
        btn.disabled = false;
      }
    });
    return h("div", {}, h("div", { class: "actions" }, btn,
      h("small", { class: "muted" }, "같은 처방전 사진으로 처음부터 다시 처리합니다. 이 처방은 삭제되고 새 처방 번호가 생깁니다.")), err);
  }

  // 6장 E_DB [저장 다시 시도]: 검수까지 끝난 처방이라 처음부터 다시 하지 않고 조제 기록 저장만 다시 한다
  function saveRetryButton(rx) {
    if (!rx.save_retryable) return null;
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const btn = h("button", { type: "button" }, icon("save"), "저장 다시 시도");
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      err.hidden = true;
      try {
        await api(`/api/prescriptions/${rx.id}/save-retry`, { method: "POST" });
        await refresh();
      } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
        btn.disabled = false;
      }
    });
    return h("div", {}, h("div", { class: "actions" }, btn,
      h("small", { class: "muted" }, "검수까지 끝난 내용으로 조제 기록 저장만 다시 합니다.")), err);
  }

  // 6장 재촬영: 실패한 처방을 새로 찍은 처방전 사진으로 다시 접수한다 (새 처방 번호, 이 처방은 삭제 = 취소됨)
  function retakeForm(rx) {
    if (!rx.retake_allowed) return null;
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const file = h("input", { type: "file", name: "image", accept: "image/jpeg,image/png", required: true,
      id: `retake-${rx.id}` });
    const btn = h("button", { type: "submit", class: rx.retryable ? "secondary" : null }, icon("upload"), "새 사진으로 다시 올리기");
    const form = h("form", { class: "upload" },
      h("label", { for: `retake-${rx.id}` }, "처방전을 다시 찍어 올리기"),
      h("div", { class: "upload-row" }, fileDrop(file), btn),
      h("small", { class: "muted" }, "새 처방 번호로 처음부터 처리하고, 이 처방은 삭제(취소됨)로 남습니다."), err);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      btn.disabled = true;
      err.hidden = true;
      try {
        const next = await api(`/api/prescriptions/${rx.id}/retake`, { method: "POST", body: new FormData(form) });
        upsertListItem(next);
        await open(next.id);
      } catch (ex) {
        err.textContent = ex.message;
        err.hidden = false;
        btn.disabled = false;
      }
    });
    return form;
  }

  function failure(rx, withLines) {
    return [h("p", { class: "form-error" }, rx.error_message || "처리 중 오류가 났습니다."),
      rx.error_code ? h("p", { class: "muted mono" }, rx.error_code) : null,
      withLines ? failedImage(rx) : null,
      retryButton(rx), saveRetryButton(rx), retakeForm(rx)];
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
      steps.push(step("error", "조제지시서 작성", rx.updated_at, ...failure(rx, true)));
    } else if (s === "cancelled") {
      steps.push(step("cancelled", "조제지시서 확인·승인", rx.updated_at));
    } else {
      steps.push(step("active", "조제지시서 작성", null, spinner("처방전을 읽고 조제지시서와 조제 계획을 만드는 중입니다…")));
    }
    const flowOpen = !["failed", "cancelled"].includes(s);

    // 트레이 검수 (F-WEB-02 ④ · F-WEB-09): 승인 뒤 트레이 사진을 올리면 검수 → 전 칸 일치면 완료, 아니면 HITL ②
    const trays = events.filter((e) => e.event === "tray_uploaded");
    const lastTray = trays[trays.length - 1];
    const inspected = events.filter((e) => e.event === "inspected");
    const trayChecking = s === "processing" && lastTray && !inspected.some((e) => e.image_id === lastTray.image_id);
    const attempt = trays.length > 1 ? ` (${trays.length}번째 사진)` : "";
    if (!approval) {
      steps.push(step(flowOpen ? "pending" : (s === "failed" ? "pending" : "cancelled"), "트레이 검수", null,
        flowOpen ? h("p", { class: "muted" }, "조제지시서를 승인하면 조제 후 트레이 사진을 올릴 수 있습니다.") : null));
    } else if (s === "waiting_tray") {
      const redone = events.filter((e) => e.event === "response" && e.hitl_point === "inspection" && e.action === "redo");
      steps.push(step("active", `트레이 검수${redone.length ? ` (재조제 ${redone.length}회)` : ""}`, null,
        trayUpload(rx, order, redone.length > 0)));
    } else if (trayChecking) {
      steps.push(step("active", `트레이 검수${attempt}`, lastTray.at, spinner("트레이 사진에서 칸별 알약을 확인하는 중입니다…")));
    } else if (s === "processing") {
      steps.push(step("active", "승인 내용 반영", null, spinner("승인한 내용으로 조제 계획을 확정하는 중입니다…")));
    } else if (s === "waiting_approval" && rx.hitl_point === "inspection" && rx.card) {
      steps.push(step("active", `트레이 검수 확인${attempt}`, inspected.length ? inspected[inspected.length - 1].at : null,
        inspectionCard(rx)));
    } else if (s === "done") {
      const last = inspected[inspected.length - 1];
      const manual = events.some((e) => e.event === "response" && e.hitl_point === "inspection" && e.action === "manual");
      steps.push(step("done", `트레이 검수${attempt}`, last && last.at, rx.card && rx.card.inspection
        ? doneInspection(rx)
        : manual ? h("p", { class: "ok-note" }, "검출 모델 오류로 약사가 육안 검수 후 수동으로 진행했습니다.") : null));
    } else if (s === "failed") {
      steps.push(step("error", "승인 이후 처리", rx.updated_at, ...failure(rx)));
    } else if (s === "cancelled") {
      steps.push(step("cancelled", "트레이 검수", null));
    }

    // 마무리
    if (s === "done") steps.push(step("done", "조제 완료", rx.updated_at));
    else if (s === "cancelled") steps.push(step("cancelled", "취소됨", rx.updated_at));
    else if (s !== "failed") steps.push(step("pending", "조제 완료", null));

    let stage, state;
    if (s === "done") [stage, state] = [4, "done"];
    else if (!approval) {
      stage = s === "waiting_approval" ? 1 : 0;                       // 처리 중이면 인식 단계, 승인 대기면 승인 단계
      state = s === "failed" ? "error" : s === "cancelled" ? "cancelled" : "active";
    } else {
      stage = 2;                                                        // 승인 뒤는 모두 트레이 검수 단계
      state = s === "failed" ? "error" : s === "cancelled" ? "cancelled" : "active";
    }
    return h("div", {}, header(rx), progress(stage, state), h("ol", { class: "timeline" }, steps));
  }

  // onSelect가 있으면 줄을 누르거나 Enter로 고를 수 있다 (F-WEB-06 원문 위치 표시)
  function itemsTable(order, lines, onSelect, changed) {
    const changedSet = new Set(changed || []);   // Q7: 약사가 고친 뒤 다시 검수한 줄 번호(0부터)
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
          // 약을 사전에서 정해 표준 이름으로 바꾼 줄: 처방전에서 읽은 글자도 보인다 (Agent printed_name, 11/23 [N])
          it.printed_name ? h("div", { class: "muted printed-name" }, `인식한 글자: ${it.printed_name}`) : null,
          changedSet.has(idx) ? h("div", {}, h("span", { class: "chip caution" }, "수정 후 다시 검수")) : null,
          ...(it.check_reasons || []).map((r) => h("div", { class: "reason" }, r))),
        h("td", { class: "num" }, it.count_per_dose),
        h("td", {}, times),
        h("td", { class: "num" }, `${it.days}일`),
        h("td", {}, low ? h("span", { class: "chip low" }, "낮음")
          : (confs.length ? h("span", { class: "chip ok" }, "정상")
            : h("span", { class: "muted", title: "처방전에서 읽은 위치 정보가 없어 신뢰도를 알 수 없습니다" }, "-"))));
      if (onSelect) {
        tr.addEventListener("click", () => onSelect(idx, tr));
        tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(idx, tr); } });
      }
      return tr;
    });
    return h("div", { class: "table-wrap" }, h("table", { class: "items" },
      h("thead", {}, h("tr", {}, h("th", {}, "약품"), h("th", {}, "1회"), h("th", {}, "복용 시점"),
        h("th", {}, "일수"), h("th", {}, "인식 신뢰도"))),
      h("tbody", {}, rows)));
  }

  // 조제지시서 warnings(A.5): 처방전을 읽는 중 Agent가 알린 내용. 복용 시점 누락·종이 잘림·API 미사용 등.
  function warningsPanel(warnings) {
    if (!warnings || !warnings.length) return null;
    return h("div", { class: "risk risk-warning", role: "status" },
      h("strong", {}, "처방전을 읽을 때 알린 내용"),
      h("ul", {}, warnings.map((w) => h("li", {}, w))));
  }

  function riskPanel(risk) {
    const level = risk.level || "none";
    const box = h("div", { class: `risk risk-${level}` },
      h("div", { class: "risk-head" }, h("strong", {}, `위험 신호: ${RISK[level]}`)));   // 판단 범위(처방약 기준)는 화면에 보이지 않는다 (11/26)
    if (risk.findings && risk.findings.length) {
      // 규칙 문장(굵게) · AI 설명(이름표 붙은 따로 된 칸) · 근거를 줄을 나눠 보인다 (11/26 가독성)
      box.append(h("ul", { class: "findings" }, risk.findings.map((f) =>
        h("li", { class: "finding" },
          h("div", { class: "finding-msg" }, h("span", { class: `chip ${f.level}` }, RISK[f.level] || f.level), " ", f.message),
          f.explanation ? h("div", { class: "finding-ai" }, h("span", { class: "finding-ai-label" }, "AI 설명"), f.explanation) : null,
          evidenceList(f.evidence)))));
    }
    return box;
  }

  // 근거 출처 "식약처 의약품 제품 허가정보 (https://...)" → 이름과 주소로 나눈다 (주소는 링크로만 보인다)
  function sourceParts(source) {
    const m = /^(.*?)\s*\((https?:\/\/[^\s)]+)\)\s*$/.exec(source || "");
    return m ? { name: m[1], url: m[2] } : { name: source || "", url: null };
  }

  // 위험 항목의 대조 근거(A.5 risk_check.findings[].evidence, RAG). 접어 두고 펼쳐 본다.
  // 근거를 못 찾았으면 '근거 없음'(기능분석서 F-AG-06: 위험 단계는 DUR 규칙으로 유지, 흐름은 계속). 이력 상세와 같은 표시.
  function evidenceList(evidence) {
    if (!evidence || !evidence.length) return h("div", { class: "muted evidence-none" }, "근거 없음");
    return h("details", { class: "evidence" },
      h("summary", {}, `근거 ${evidence.length}건 보기`),
      h("ul", {}, evidence.map((ev) => {
        const src = sourceParts(ev.source);
        return h("li", {},
          h("div", { class: "ev-head" }, h("strong", {}, ev.title || ""),
            src.url ? h("a", { href: src.url, target: "_blank", rel: "noopener" }, `${src.name || "출처"} ↗`)
              : h("span", { class: "muted" }, src.name)),
          h("blockquote", { class: "ev-excerpt" }, ev.excerpt || ""));
      })));
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
      h("table", { class: "items plan" }, h("thead", {}, h("tr", {}, h("th", {}, "일"), h("th", {}, "시간"), h("th", {}, "넣을 약"))),
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
        h("td", {}, h("button", { type: "button", class: "link", onclick: () => tr.remove(), "aria-label": "이 항목 삭제" }, icon("x"), "삭제")));
      return tr;
    }
    order.items.forEach((it, i) => body.append(row(it, i)));
    const add = h("button", { type: "button", class: "link", onclick: () => body.append(row(null, null)) }, icon("plus"), "항목 추가");
    const send = h("button", { type: "button" }, icon("calc"), "수정 내용으로 다시 계산");
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
    const cancel = h("button", { type: "button", class: "link", onclick: onClose }, icon("x"), "수정 취소");
    return h("div", { class: "edit-form" },
      h("p", { class: "muted" }, "이름을 바꾸면 약품을 다시 찾고, 조제 계획과 위험 신호도 다시 계산합니다."),
      h("table", { class: "items" },
        h("thead", {}, h("tr", {}, ["약품", "1회 알 수", "복용 시간", "일수", ""].map((t) => h("th", {}, t)))), body),
      h("div", { class: "actions" }, add, cancel, send), err);
  }

  function approvalCard(rx) {
    const { order, plan, notice, ocr_lines: lines, ocr_size: size, blockers, needs_confirm: needsConfirm, is_danger: isDanger } = rx.card;
    const changed = rx.card.changed_items || [];
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
      icon("check"), isDanger ? "사유 입력 후 진행" : "조제지시서 승인");
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
    // OCR 좌표는 펴진 처방전 이미지 기준이라 박스를 그릴 때는 그 이미지를 보여 준다 (원본은 [원본 보기])
    const shownUrl = rx.image_url && canLocate ? `${rx.image_url}?view=rect` : rx.image_url;
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
        rx.image_url ? h("figure", { class: "rx-image",
          // 처방전 그림이나 [확대]를 눌렀을 때만 크게 본다 (아래 빈 곳은 반응하지 않음). 크게 본 상태에서는 아무 곳이나 누르면 닫힌다
          onclick: (e) => {
            const fig = e.currentTarget;
            if (e.target.closest("a")) return;
            if (fig.classList.contains("zoomed")) fig.classList.remove("zoomed");
            else if (e.target.closest(".img-wrap, .zoom-btn")) fig.classList.add("zoomed");
          },
          onkeydown: (e) => {
            if (e.key === "Escape") e.currentTarget.classList.remove("zoomed");
            else if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("img-wrap")) {
              e.preventDefault();
              e.currentTarget.classList.toggle("zoomed");
            }
          } },
          h("div", { class: "img-wrap", role: "button", tabindex: "0", "aria-label": "처방전 크게 보기" },
            h("img", { src: shownUrl, alt: `처방 ${rx.display_no} ${canLocate ? "펴진 처방전" : "원본"} 사진` }), boxes, zoomHint()),
          h("figcaption", { class: "img-actions" },
            h("button", { type: "button", class: "chip-link zoom-btn" }, icon("zoom"), "확대"),
            canLocate ? originalLink(rx.image_url) : null)) : null,
        h("div", { class: "approval-body" },
          h("p", { class: "muted" }, canLocate
            ? "항목을 누르면 처방전에서 읽은 위치를 표시합니다. 노란 줄은 인식 신뢰도가 낮거나 확인이 필요한 항목입니다."
            : "원본과 대조한 뒤 승인하세요."),
          warningsPanel(order.warnings),
          changed.length ? h("p", { class: "muted" },
            `수정한 줄 중 ${changed.length}줄(${changed.map((i) => i + 1).join(", ")}번째)을 다시 검수했습니다. 나머지 줄의 검수 결과는 그대로입니다.`) : null,
          itemsTable(order, lines, canLocate && rx.image_url ? select : null, changed),
          planPanel(order, plan, notice),
          riskPanel(order.risk_check || {}))),
      blockers.map((b) => h("p", { class: "form-error" }, b)),
      reasonBox,
      h("div", { class: "actions" }, isDanger ? null : confirm, btn,
        h("button", { type: "button", class: "link", onclick: () => {
          const f = editForm(rx, () => { form.replaceChildren(); });
          form.replaceChildren(f);
          f.querySelector("input").focus();
        } }, icon("edit"), "항목 수정")),
      err, form);
  }

  // ---------------------------------------------------------- 트레이 검수
  function drugNames(order) {
    return Object.fromEntries(((order && order.items) || []).filter((it) => it.item_code).map((it) => [it.item_code, it.name]));
  }

  function counts(c, names) {
    const shown = Object.entries(c || {}).filter(([, n]) => n).map(([code, n]) => `${names[code] || code} ${n}정`);
    return shown.length ? shown.join(", ") : "없음";
  }

  // 트레이 사진 올리기 (승인된 처방에만). 트레이를 못 찾았으면 이유를 보이고 다시 올리게 한다
  function trayUpload(rx, order, redone) {
    const retake = rx.card && rx.card.retake;
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const file = h("input", { type: "file", name: "image", accept: "image/jpeg,image/png", required: true,
      id: "tray-image", "aria-describedby": retake ? null : "tray-help" });
    const btn = h("button", { type: "submit" }, icon("upload"), retake ? "다시 올리기" : "트레이 사진 올리기");
    const form = h("form", { class: "upload" },
      h("label", { for: "tray-image", class: "sr-only" }, "트레이 사진"),
      h("div", { class: "upload-row" }, fileDrop(file, "트레이 전체가 보이게 찍은 사진을 고르거나 끌어다 놓으세요 · JPG · PNG, 10MB 이하"), btn), err);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      err.hidden = true;
      btn.disabled = true;
      setLabel(btn, null, "올리는 중…");
      try {
        await api(`/api/prescriptions/${rx.id}/tray`, { method: "POST", body: new FormData(form) });
        await refresh();
      } catch (ex) {
        err.textContent = ex.message;
        err.hidden = false;
        btn.disabled = false;
        setLabel(btn, "upload", retake ? "다시 올리기" : "트레이 사진 올리기");
      }
    });
    return h("div", { class: "tray-upload" },
      retake
        ? h("div", { class: "retake", role: "alert" }, h("strong", {}, "다시 찍어 주세요"), h("p", {}, retake.message),
          retake.code ? h("p", { class: "muted mono" }, retake.code) : null)
        : h("p", { class: "ok-note" }, redone
          ? "재조제: 조제 계획대로 트레이를 다시 채운 뒤 사진을 올려 주세요."
          : "조제지시서를 승인했습니다. 조제 계획대로 트레이를 채운 뒤 사진을 올려 주세요."),
      retake ? null : h("p", { class: "muted", id: "tray-help" }, "트레이 전체가 보이고 아침 줄이 위로 오게 찍어 주세요."),
      form,
      retake && retake.manual_allowed ? manualInspection(rx) : null,
      trayReference(rx, order));
  }

  // 트레이 대기 때 펼쳐 볼 것 (11/26): 처음 채울 때는 칸 배치, 재조제로 돌아왔으면 직전 검수 결과 (틀린 칸을 보고 고쳐 담게)
  // 승인한 조제지시서는 위 '조제지시서 승인' 단계에 있으므로 여기서 되풀이하지 않는다
  function trayReference(rx, order) {
    const card = rx.card || {};
    const names = drugNames(order);
    if (card.previous && card.previous.inspection) {
      const prev = { display_no: rx.display_no, card: card.previous };
      return h("details", {}, h("summary", {}, "이전 검수 결과 보기"), trayLayout(prev, card.previous.inspection, names));
    }
    if (card.plan) return h("details", {}, h("summary", {}, "칸 배치 보기"), planGrid(card.plan, names));
    return null;
  }

  // 조제 계획(A.4)을 트레이 모양(4행 시간 × 3열 일)으로: 칸마다 넣을 약과 알 수
  function planGrid(plan, names) {
    const byKey = Object.fromEntries((plan.plan || []).map((c) => [`${c.day}-${c.time}`, c]));
    const body = TIME_ORDER.map((t) => h("tr", {}, h("th", { scope: "row" }, TIMES[t]), DAYS.map((d) => {
      const c = byKey[`${d}-${t}`];
      const items = (c && c.items) || [];
      return h("td", {}, h("div", { class: `cell plan-cell${items.length ? "" : " cell-empty"}` },
        items.length ? items.map((i) => h("span", { class: "plan-item" }, `${names[i.item_code] || i.item_code} ${i.count}정`))
          : h("span", { class: "muted" }, d > (plan.days || 3) ? "-" : "빈 칸")));
    })));
    return h("table", { class: "tray-grid plan-grid" },
      h("thead", {}, h("tr", {}, h("th", {}, ""), DAYS.map((d) => h("th", { scope: "col" }, `${d}일`)))),
      h("tbody", {}, body));
  }

  // 6장 E_CV_INFER: 검출 모델 오류로 자동 검수를 못 할 때 약사가 트레이를 직접 보고 사유를 적어 완료한다
  function manualInspection(rx) {
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const reason = h("textarea", { class: "reason-input", rows: 2, maxlength: 500, id: `manual-${rx.id}`,
      placeholder: "예: 조제 계획과 12칸을 육안으로 대조함, 이상 없음" });
    const btn = h("button", { type: "button", class: "secondary" }, icon("eye"), "육안 검수로 완료");
    btn.addEventListener("click", async () => {
      err.hidden = true;
      if (!reason.value.trim()) {
        err.textContent = "육안으로 확인한 내용(사유)을 적어 주세요.";
        err.hidden = false;
        return;
      }
      if (!window.confirm("자동 검수 없이 육안 검수로 완료할까요? 조제 기록에 검수 결과 없이 사유가 남습니다.")) return;
      btn.disabled = true;
      try {
        await api(`/api/prescriptions/${rx.id}/manual-inspection`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: reason.value }) });
        await refresh();
      } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
        btn.disabled = false;
      }
    });
    return h("details", { class: "manual-inspection" },
      h("summary", {}, "계속 실패하면: 육안 검수 후 수동 진행"),
      h("p", { class: "muted" }, "조제 계획 표와 트레이 12칸을 직접 대조한 뒤 확인한 내용을 적어 주세요."),
      h("label", { for: `manual-${rx.id}` }, "확인 내용(사유)"), reason,
      h("div", { class: "actions" }, btn), err);
  }

  // 4행(아침·점심·저녁·취침 전) × 3열(1~3일) 격자. 칸마다 판정 글자를 함께 써서 색만으로 구분하지 않는다.
  // 칸을 누르면 예상·실제 약과 확인 필요 사유를 아래에 보인다 (F-WEB-09 ②)
  function trayGrid(a3, names, detail) {
    const byKey = Object.fromEntries((a3.cells || []).map((c) => [`${c.day}-${c.time}`, c]));
    function show(c, btn) {
      for (const b of btn.closest("table").querySelectorAll(".cell")) b.toggleAttribute("aria-pressed", b === btn);
      detail.replaceChildren(
        h("strong", {}, `${c.day}일 ${TIMES[c.time]} — ${CELL[c.status] || c.status}`),
        h("dl", {},
          h("dt", {}, "예상"), h("dd", {}, counts(c.expected, names)),
          h("dt", {}, "실제"), h("dd", {}, counts(c.actual, names)),
          c.reasons && c.reasons.length ? [h("dt", {}, "사유"), h("dd", {}, c.reasons.map((r) => CELL_REASON[r] || r).join(", "))] : null));
    }
    const body = TIME_ORDER.map((t) => h("tr", {}, h("th", { scope: "row" }, TIMES[t]), DAYS.map((d) => {
      const c = byKey[`${d}-${t}`];
      if (!c) return h("td", {}, h("span", { class: "muted" }, "-"));
      const empty = !Object.values(c.expected || {}).some(Boolean) && !Object.values(c.actual || {}).some(Boolean);
      const btn = h("button", { type: "button", class: `cell cell-${c.status}${empty ? " cell-empty" : ""}`,
        "aria-label": `${d}일 ${TIMES[t]} ${CELL[c.status]}` },
        h("span", { class: "cell-state" }, empty && c.status === "ok" ? "빈 칸" : CELL[c.status]),
        h("span", { class: "cell-items" }, empty ? "" : counts(c.actual, names)));
      btn.addEventListener("click", () => show(c, btn));
      return h("td", {}, btn);
    })));
    const grid = h("table", { class: "tray-grid" },
      h("thead", {}, h("tr", {}, h("th", {}, ""), DAYS.map((d) => h("th", { scope: "col" }, `${d}일`)))),
      h("tbody", {}, body));
    const first = (a3.cells || []).find((c) => c.status !== "ok");
    if (first) {   // 문제가 있는 첫 칸을 미리 펼쳐 둔다
      const btn = grid.querySelector(`tbody tr:nth-child(${TIME_ORDER.indexOf(first.time) + 1}) td:nth-child(${first.day + 1}) .cell`);
      if (btn) show(first, btn);
    } else {
      detail.replaceChildren(h("span", { class: "muted" }, "칸을 누르면 예상·실제 약을 보여 줍니다."));
    }
    return grid;
  }

  function summaryLine(a3) {
    const s = a3.summary || {};
    return h("p", { class: "tray-summary" },
      h("span", { class: "chip ok" }, `일치 ${s.ok || 0}`), " ",
      h("span", { class: "chip danger" }, `불일치 ${s.mismatch || 0}`), " ",
      h("span", { class: "chip warning" }, `확인 필요 ${s.need_review || 0}`));
  }

  function trayLayout(rx, a3, names, extra) {
    const detail = h("div", { class: "cell-detail", "aria-live": "polite" });
    const url = rx.card.tray_image_url;
    // 검수 결과 그림(보정한 트레이 위 칸 판정 색·검출 박스, 11/23 [K])이 있으면 그것을 보이고 원본은 링크로 둔다
    const overlay = rx.card.tray_overlay_url;
    return h("div", { class: "approval-grid" },
      overlay ? h("figure", { class: "rx-image" },
        h("a", { href: overlay, target: "_blank", rel: "noopener", class: "img-wrap" },
          h("img", { src: overlay, alt: `처방 ${rx.display_no} 트레이 검수 결과 그림` }), zoomHint()),
        url ? h("figcaption", { class: "img-actions" }, originalLink(url)) : null) :
      url ? h("figure", { class: "rx-image" },
        h("a", { href: url, target: "_blank", rel: "noopener", class: "img-wrap" },
          h("img", { src: url, alt: `처방 ${rx.display_no} 트레이 사진` }), zoomHint())) : null,
      h("div", { class: "approval-body" }, summaryLine(a3), trayGrid(a3, names, detail), detail, ...(extra || [])));
  }

  // HITL ② 검수 결과 카드: 불일치·확인 필요 칸이 있을 때만 온다 (전 칸 일치면 바로 완료)
  function inspectionCard(rx) {
    const { inspection: a3, names, notice } = rx.card;
    const extra = [notice ? h("p", { class: "retake", role: "status" }, notice) : null];
    // HITL ② 응답 (F-WEB-09 ③): 확인 완료 → 완료, 재조제 → 트레이를 다시 채워 사진을 다시 올린다
    const err = h("p", { class: "form-error", role: "alert", hidden: true });
    const redo = h("button", { type: "button", class: "secondary" }, icon("tray"), "재조제 (트레이 다시 채우기)");
    const ok = h("button", { type: "button" }, icon("check"), "확인 완료");
    async function reply(action) {
      err.hidden = true;
      redo.disabled = ok.disabled = true;
      try {
        await api(`/api/prescriptions/${rx.id}/inspection`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        await refresh();
      } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
        redo.disabled = ok.disabled = false;
      }
    }
    redo.addEventListener("click", () => reply("redo"));
    ok.addEventListener("click", () => reply("confirm"));
    return h("div", { class: "approval" }, trayLayout(rx, a3, names || {}, extra),
      h("div", { class: "actions" }, redo, ok), err);
  }

  function doneInspection(rx) {
    const a3 = rx.card.inspection;
    const allOk = !(a3.summary && a3.summary.hitl_required);   // 아니면 HITL ②에서 약사가 확인 완료를 누른 것
    return h("div", { class: "approval" },
      h("p", { class: "ok-note" }, allOk ? "전 칸 일치 — 약사 확인 없이 완료했습니다." : "약사가 검수 결과를 확인했습니다."),
      h("details", {}, h("summary", {}, "검수 결과 보기"), trayLayout(rx, a3, drugNames(rx.card.order))));
  }

  // 새로고침해도 보던 처방을 다시 연다 (REQ-WEB-03)
  // 주소에 처방이 없으면 새 처방 접수 화면으로 시작한다 (예전처럼 최근 처방을 열면 올린 사진이 그 처방에 붙는 것처럼 보였다)
  const m = location.hash.match(/^#rx=([0-9a-f-]{36})$/);
  if (m) open(m[1]);
})();
