/* Progressive enhancement only: no network, storage, tracking, or real tasks. */
(() => {
  const picker = document.querySelector('.platform-picker');
  const tabs = Array.from(picker.querySelectorAll('[data-platform]'));
  const panels = tabs.map(tab => document.getElementById(`panel-${tab.dataset.platform}`));
  picker.setAttribute('role', 'tablist');
  function select(index, focus = false) {
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
    });
    if (focus) tabs[index].focus();
  }
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].tabIndex = 0;
    tab.addEventListener('click', event => { event.preventDefault(); select(index); });
    tab.addEventListener('keydown', event => {
      const next = {ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1}[event.key];
      if (next !== undefined) { event.preventDefault(); select(next, true); }
      else if (event.key === ' ') { event.preventDefault(); select(index, true); }
    });
  });
  function selectHash() {
    const index = panels.findIndex(panel => `#${panel.id}` === location.hash);
    if (index >= 0) select(index);
    return index;
  }
  if (selectHash() < 0) select(0);
  window.addEventListener('hashchange', selectHash);
})();

/* Lightweight entry animations; content stays visible without JS or motion. */
(() => {
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (!motion.matches && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.remove('reveal-pending');
          observer.unobserve(entry.target);
        }
      });
    }, {threshold: 0.08});
    document.querySelectorAll('[data-reveal]').forEach(element => {
      if (element.getBoundingClientRect().top >= window.innerHeight) {
        element.classList.add('reveal-pending');
        observer.observe(element);
      }
    });
    motion.addEventListener('change', event => {
      if (event.matches) {
        observer.disconnect();
        document.querySelectorAll('.reveal-pending').forEach(element => element.classList.remove('reveal-pending'));
      }
    });
  }
})();

/* Local static QR: the destination is the public page, never a login token. */
(() => {
  const widget = document.querySelector('.qr-widget');
  const toggle = widget.querySelector('.qr-toggle');
  const panel = widget.querySelector('.qr-panel');
  const close = widget.querySelector('.qr-close');
  const desktop = window.matchMedia('(min-width: 1101px)');
  function setOpen(open, focus = false) {
    panel.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    if (focus) (open ? close : toggle).focus();
  }
  widget.hidden = false;
  setOpen(desktop.matches && window.scrollY < 120);
  // The introductory card yields to the gallery when reading begins.
  function collapseIntro() {
    if (window.scrollY > 120) {
      setOpen(false);
      window.removeEventListener('scroll', collapseIntro);
    }
  }
  window.addEventListener('scroll', collapseIntro, {passive: true});
  toggle.addEventListener('click', () => setOpen(panel.hidden, true));
  close.addEventListener('click', () => setOpen(false, true));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !panel.hidden) {
      const focusInside = widget.contains(document.activeElement);
      setOpen(false, focusInside);
    }
  });
  document.addEventListener('click', event => {
    if (!panel.hidden && !widget.contains(event.target)) setOpen(false);
  });
  desktop.addEventListener('change', () => setOpen(false));
})();

/* A bounded hero expansion follows native scrolling; never intercept input. */
(() => {
  const stage = document.querySelector('.hero-stage');
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const links = Array.from(document.querySelectorAll('.nav nav a[href^="#"]'));
  let scheduled = false;
  function render() {
    scheduled = false;
    const rect = stage.getBoundingClientRect();
    const progress = Math.max(0, Math.min(1, (innerHeight - rect.top) / (innerHeight * .75)));
    stage.style.setProperty('--stage-scale', motion.matches ? '1' : String(.965 + .035 * progress));
    let current = null;
    links.forEach(link => {
      const section = document.querySelector(link.getAttribute('href'));
      if (section && section.getBoundingClientRect().top <= 150) current = link;
    });
    links.forEach(link => {
      if (link === current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  function requestRender() {
    if (!scheduled) { scheduled = true; requestAnimationFrame(render); }
  }
  window.addEventListener('scroll', requestRender, {passive: true});
  window.addEventListener('resize', requestRender);
  motion.addEventListener('change', requestRender);
  render();
})();

/* Mobile scenes autoplay only while visible; desktop scenes remain reader-controlled.
   Manual browsing pauses rotation. No timers or animation with reduced motion. */
document.querySelectorAll('[data-gallery]:not(.mobile-gallery)').forEach(gallery => {
  const list = gallery.querySelector('.gallery-tabs');
  const tabs = Array.from(gallery.querySelectorAll('[data-gallery-tab]'));
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('href').slice(1)));
  const live = gallery.querySelector('.gallery-live');
  const play = gallery.querySelector('[data-gallery-play]');
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const duration = Number(gallery.dataset.autoplay) || 0;
  let current = 0;
  let timer = null;
  let paused = false;
  let inView = false;
  list.setAttribute('role', 'tablist');
  gallery.classList.add('is-enhanced');

  function syncPlayback() {
    clearTimeout(timer);
    timer = null;
    gallery.classList.remove('is-playing');
    if (!play) return;
    play.disabled = motion.matches;
    play.textContent = motion.matches ? '静态' : paused ? '播放' : '暂停';
    play.setAttribute('aria-label', motion.matches ? '已按减少动态效果设置停止自动播放' : paused ? '开始自动播放' : '暂停自动播放');
    if (duration && !paused && !motion.matches && !document.hidden && inView) {
      // Restart the active scene's progress when resuming; never catch up off-screen.
      void gallery.offsetWidth;
      gallery.classList.add('is-playing');
      timer = setTimeout(() => select(current + 1), duration);
    }
  }
  function select(index, focus = false, announce = false) {
    current = (index + tabs.length) % tabs.length;
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === current));
      tab.tabIndex = i === current ? 0 : -1;
      panels[i].hidden = i !== current;
    });
    if (focus) tabs[current].focus();
    if (announce) live.textContent = `${current + 1} / ${tabs.length}：${tabs[current].textContent.trim()}`;
    syncPlayback();
  }
  function manual(index, focus = false, announce = false) {
    paused = true;
    select(index, focus, announce);
  }
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].tabIndex = 0;
    tab.addEventListener('click', event => { event.preventDefault(); manual(index); });
    tab.addEventListener('keydown', event => {
      const next = {ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1}[event.key];
      if (next !== undefined) { event.preventDefault(); manual(next, true); }
      else if (event.key === ' ') { event.preventDefault(); manual(index, true); }
    });
  });
  gallery.querySelector('[data-gallery-prev]').addEventListener('click', () => manual(current - 1, false, true));
  gallery.querySelector('[data-gallery-next]').addEventListener('click', () => manual(current + 1, false, true));
  gallery.querySelector('.gallery-arrows').hidden = false;
  if (play) {
    play.addEventListener('click', () => { paused = !paused; syncPlayback(); });
    // Do not replace the scene beneath a keyboard reader. Resume is always explicit.
    gallery.addEventListener('focusin', event => {
      if (event.target !== play) { paused = true; syncPlayback(); }
    });
    document.addEventListener('visibilitychange', syncPlayback);
    motion.addEventListener('change', syncPlayback);
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(entries => {
        inView = entries[0].isIntersecting;
        syncPlayback();
      }, {threshold: 0.15}).observe(gallery);
    }
    // If visibility observation is unavailable, stay in manual mode.
  }
  function selectHash() {
    const index = panels.findIndex(panel => `#${panel.id}` === location.hash);
    if (index >= 0) manual(index);
    return index;
  }
  if (selectHash() < 0) select(0);
  window.addEventListener('hashchange', selectHash);
});

/* A white, automatic visual story. No mode switcher; reduced motion shows all scenes. */
(() => {
  const gallery = document.querySelector('.mobile-gallery');
  const panels = Array.from(gallery.querySelectorAll('.gallery-panel'));
  const play = gallery.querySelector('[data-gallery-play]');
  const controls = gallery.querySelector('.showcase-controls');
  if (!('IntersectionObserver' in window)) return;
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let current = 0, timer = null, paused = false, visible = false;
  function sync() {
    clearTimeout(timer);
    gallery.classList.toggle('is-playing', !paused && visible && !document.hidden && !motion.matches);
    gallery.classList.toggle('is-enhanced', !motion.matches);
    controls.hidden = motion.matches;
    panels.forEach((panel, i) => { panel.hidden = !motion.matches && i !== current; });
    play.textContent = paused ? '播放动画' : '暂停动画';
    play.setAttribute('aria-label', paused ? '播放动态演示' : '暂停动态演示');
    play.setAttribute('aria-pressed', String(paused));
    if (!paused && visible && !document.hidden && !motion.matches) {
      timer = setTimeout(() => { current = (current + 1) % panels.length; sync(); }, 6500);
    }
  }
  function hash() {
    const index = panels.findIndex(panel => '#' + panel.id === location.hash);
    if (index >= 0) { current = index; paused = true; }
    sync();
  }
  play.addEventListener('click', () => { paused = !paused; sync(); });
  gallery.addEventListener('focusin', event => { if (event.target !== play) { paused = true; sync(); } });
  motion.addEventListener('change', sync);
  document.addEventListener('visibilitychange', sync);
  window.addEventListener('hashchange', hash);
  if ('IntersectionObserver' in window) new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting; sync();
  }, {threshold: .15}).observe(gallery);
  hash();
})();
