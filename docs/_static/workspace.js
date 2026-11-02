/* Progressive enhancement only: all chapter, outline and API links work without JS. */
document.addEventListener('DOMContentLoaded', () => {
  document.documentElement.classList.add('sw-enhanced');
  const zh = document.documentElement.lang === 'zh-CN';
  // RTD scrollIntoView can put the selected chapter beneath the sticky search.
  // Keep global navigation stable; section jumps belong to the local outline.
  const resetGlobalScroll = () => requestAnimationFrame(() => {
    const pane = document.querySelector('.wy-side-scroll');
    if (pane) pane.scrollTop = 0;
  });
  window.addEventListener('load', resetGlobalScroll);
  window.addEventListener('hashchange', resetGlobalScroll);
  const outline = document.querySelector('.sw-outline details');
  if (outline && matchMedia('(max-width: 1200px)').matches) outline.open = false;
  if (outline) outline.addEventListener('click', event => {
    if (event.target.closest('a[href^="#"]') && matchMedia('(max-width: 1200px)').matches) outline.open = false;
  });
  const links = [...document.querySelectorAll('.sw-outline nav a[href^="#"]')];
  const refresh = () => links.forEach(a => {
    if (a.hash === location.hash) a.setAttribute('aria-current', 'location');
    else a.removeAttribute('aria-current');
  });
  addEventListener('hashchange', refresh); refresh();
  const usage = document.querySelector('#usage');
  if (!usage) return;
  const entries = [...usage.querySelectorAll('dt.sig[id]')].map(node => ({id:node.id, name:node.id}));
  if (!entries.length) return;
  const finder = document.createElement('section'); finder.className = 'sw-api-finder';
  const label = document.createElement('label'); label.htmlFor = 'sw-api-query';
  label.textContent = zh ? '查找本页 API' : 'Find an API on this page';
  const input = document.createElement('input'); input.id = 'sw-api-query'; input.type = 'search';
  input.placeholder = zh ? '输入函数名，例如 scalar、vti、elastic…' : 'Filter by name: scalar, vti, elastic…';
  input.setAttribute('aria-controls', 'sw-api-results');
  const results = document.createElement('nav'); results.id = 'sw-api-results'; results.className = 'sw-api-results';
  results.setAttribute('aria-label', zh ? 'API 快捷链接' : 'API quick links');
  const status = document.createElement('p'); status.className = 'sw-api-status'; status.setAttribute('role','status');
  const render = () => {
    const query = input.value.trim().toLowerCase();
    const selected = entries.filter(e => e.name.toLowerCase().includes(query));
    results.replaceChildren(...selected.map(e => { const a = document.createElement('a'); a.href = '#' + encodeURIComponent(e.id); a.textContent = e.name; return a; }));
    status.textContent = selected.length ? (zh ? `${selected.length} 个匹配接口 · 仅筛选快捷链接，正文保持完整` : `${selected.length} matching APIs · filters links, not documentation`) : (zh ? '没有匹配项。尝试其他名称；全部正文仍在下方。' : 'No matches. Try another name; the full reference remains below.');
  };
  input.addEventListener('input',render);
  finder.append(label,input,results,status); usage.querySelector('h1').after(finder); render();
});
