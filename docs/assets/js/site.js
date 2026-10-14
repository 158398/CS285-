/* CS285 教程站交互:进度条 / scrollspy / tabs / lightbox / 复制 / 返回顶部 / journey */
(function () {
  // 阅读进度条
  const bar = document.querySelector(".progress");
  const onScroll = () => {
    const h = document.documentElement;
    const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
    if (bar) bar.style.width = (p * 100).toFixed(2) + "%";
    const totop = document.querySelector(".totop");
    if (totop) totop.classList.toggle("show", h.scrollTop > 500);
    // scrollspy
    const links = document.querySelectorAll(".navbar nav a[href^='#']");
    if (links.length) {
      let cur = null;
      document.querySelectorAll("[data-spy]").forEach(sec => {
        if (sec.getBoundingClientRect().top <= 120) cur = sec.id;
      });
      links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + cur));
    }
  };
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // tabs
  document.querySelectorAll("[data-tabs]").forEach(root => {
    const btns = root.querySelectorAll(".tab-btn");
    const panels = root.querySelectorAll(".tab-panel");
    btns.forEach(btn => btn.addEventListener("click", () => {
      btns.forEach(b => b.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = root.querySelector("#" + btn.dataset.tab);
      if (panel) panel.classList.add("active");
      history.replaceState(null, "", "#" + btn.dataset.tab);
    }));
    // 从 hash 恢复
    const hash = location.hash.slice(1);
    if (hash && root.querySelector("#" + hash)) {
      btns.forEach(b => b.classList.toggle("active", b.dataset.tab === hash));
      panels.forEach(p => p.classList.toggle("active", p.id === hash));
    }
  });

  // lightbox
  const lb = document.createElement("div");
  lb.className = "lightbox";
  lb.innerHTML = "<img alt='zoom'>";
  document.body.appendChild(lb);
  lb.addEventListener("click", () => lb.classList.remove("open"));
  document.querySelectorAll("figure img, .zoomable").forEach(img => {
    img.addEventListener("click", () => {
      lb.querySelector("img").src = img.src;
      lb.classList.add("open");
    });
  });

  // 代码复制
  document.querySelectorAll("pre").forEach(pre => {
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "复制";
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(pre.innerText.replace(/^复制\n/, "")).then(() => {
        btn.textContent = "已复制";
        setTimeout(() => (btn.textContent = "复制"), 1200);
      });
    });
    pre.appendChild(btn);
  });

  // 返回顶部
  const totop = document.createElement("button");
  totop.className = "totop";
  totop.textContent = "↑";
  totop.title = "回到顶部";
  totop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  document.body.appendChild(totop);

  // journey stepper:点击节点 → 滚动到目标卡并高亮
  document.querySelectorAll(".jnode").forEach(node => {
    node.addEventListener("click", () => {
      document.querySelectorAll(".jnode").forEach(n => n.classList.remove("active"));
      node.classList.add("active");
      const target = document.querySelector(node.dataset.target);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.style.transition = "box-shadow .3s";
        target.style.boxShadow = "0 0 0 3px var(--amber)";
        setTimeout(() => (target.style.boxShadow = ""), 1400);
      }
    });
  });
})();
