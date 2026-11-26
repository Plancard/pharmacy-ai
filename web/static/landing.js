// 로그인 화면 기능 소개 슬라이드 (11/26 화면 디자인, 기능분석서 [S]).
// - 사진은 data-base 뒤에 .jpg → .png → .webp 순서로 찾는다. 셋 다 없으면 자리 표시(파일 이름)를 보인다.
// - 기능 소개 · 사용 방법의 화면 캡처도 같은 방식으로 찾는다.
// - 6초마다 다음 장으로 넘어간다. 마우스를 올리거나 포커스가 있으면 멈추고, 움직임 줄이기 설정이면 자동으로 넘기지 않는다.
(() => {
  "use strict";
  const EXTS = [".jpg", ".png", ".webp"];

  document.querySelectorAll("img[data-base]").forEach((img) => {
    let n = 0;
    const missing = img.parentElement.querySelector(".shot-missing");
    const next = () => {
      if (n >= EXTS.length) {
        img.hidden = true;
        if (missing) missing.hidden = false;
        return;
      }
      img.src = img.dataset.base + EXTS[n++];
    };
    img.addEventListener("error", next);
    next();
  });

  const root = document.querySelector(".slides");
  if (!root) return;
  const slides = [...root.querySelectorAll(".slide")];
  const tabs = [...root.querySelectorAll(".slide-tab")];
  let current = 0;
  let timer = null;

  function show(i) {
    current = (i + slides.length) % slides.length;
    slides.forEach((s, k) => { s.hidden = k !== current; });
    tabs.forEach((t, k) => t.setAttribute("aria-selected", k === current ? "true" : "false"));
  }

  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const interval = Number(root.dataset.interval) || 6000;
  function start() { if (!reduce && !timer) timer = setInterval(() => show(current + 1), interval); }
  function stop() { clearInterval(timer); timer = null; }

  tabs.forEach((t, k) => t.addEventListener("click", () => { show(k); stop(); start(); }));
  root.querySelector(".slide-nav.prev").addEventListener("click", () => { show(current - 1); stop(); start(); });
  root.querySelector(".slide-nav.next").addEventListener("click", () => { show(current + 1); stop(); start(); });
  root.addEventListener("mouseenter", stop);
  root.addEventListener("mouseleave", start);
  root.addEventListener("focusin", stop);
  root.addEventListener("focusout", start);
  // 기능 소개의 '탐지 전후 보기': 그 기능의 슬라이드를 띄우고 슬라이드로 올라간다
  document.querySelectorAll("[data-goto]").forEach((a) => a.addEventListener("click", () => {
    show(Number(a.dataset.goto)); stop(); start();
  }));
  show(0);
  start();
})();
