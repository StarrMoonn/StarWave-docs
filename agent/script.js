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

/* No autoplay: readers control each illustrated client scene. Without JS all remain visible. */
document.querySelectorAll('[data-gallery]').forEach(gallery => {
  const list = gallery.querySelector('.gallery-tabs');
  const tabs = Array.from(gallery.querySelectorAll('[data-gallery-tab]'));
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('href').slice(1)));
  const live = gallery.querySelector('.gallery-live');
  let current = 0;
  list.setAttribute('role', 'tablist');
  function select(index, focus = false, announce = false) {
    current = (index + tabs.length) % tabs.length;
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === current));
      tab.tabIndex = i === current ? 0 : -1;
      panels[i].hidden = i !== current;
    });
    if (focus) tabs[current].focus();
    if (announce) live.textContent = `${current + 1} / ${tabs.length}：${tabs[current].textContent.trim()}`;
  }
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].tabIndex = 0;
    tab.addEventListener('click', event => { event.preventDefault(); select(index); });
    tab.addEventListener('keydown', event => {
      const next = {ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1}[event.key];
      if (next !== undefined) { event.preventDefault(); select(next, true); }
      else if (event.key === ' ') { event.preventDefault(); select(index, true); }
    });
  });
  gallery.querySelector('[data-gallery-prev]').addEventListener('click', () => select(current - 1, false, true));
  gallery.querySelector('[data-gallery-next]').addEventListener('click', () => select(current + 1, false, true));
  gallery.querySelector('.gallery-arrows').hidden = false;
  function selectHash() {
    const index = panels.findIndex(panel => `#${panel.id}` === location.hash);
    if (index >= 0) select(index);
    return index;
  }
  if (selectHash() < 0) select(0);
  window.addEventListener('hashchange', selectHash);
});
