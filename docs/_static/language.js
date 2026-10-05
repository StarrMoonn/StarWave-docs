/* Fill the search-status strings missing from Sphinx 8.2's Chinese catalog. */
if (document.documentElement.lang === 'zh-CN' && typeof Documentation !== 'undefined') {
  Documentation.addTranslations({
    locale: 'zh_Hans_CN',
    plural_expr: '0',
    messages: {
      'Search finished, found one page matching the search query.': ['搜索完成，找到 ${resultCount} 个匹配页面。'],
      'Preparing search...': '正在准备搜索…',
      'Searching': '正在搜索',
      'Search Results': '搜索结果',
      "Your search did not match any documents. Please make sure that all words are spelled correctly and that you've selected enough categories.": '搜索未找到匹配页面。请检查拼写或尝试其它关键词。'
    }
  });
}

/* Same-page locale switching, including nested API sections and search queries. */
document.addEventListener('DOMContentLoaded', () => {
  const sidebar = document.querySelector('.wy-nav-side');
  const menu = document.querySelector('[data-sw-mobile-menu]');
  if (sidebar && menu) {
    sidebar.id = 'sw-sidebar';
    const updateMenuState = () => menu.setAttribute('aria-expanded', String(sidebar.classList.contains('shift')));
    new MutationObserver(updateMenuState).observe(sidebar, {attributes: true, attributeFilter: ['class']});
    updateMenuState();
  }
  function destination(link) {
    const target = new URL(link.getAttribute('href'), window.location.href);
    target.search = window.location.search;
    target.hash = window.location.hash;
    if (window.location.hash) {
      let id;
      try { id = decodeURIComponent(window.location.hash.slice(1)); }
      catch (_) { return target.href; }
      const element = document.getElementById(id);
      // Python object anchors are identical in both independently built locales.
      if (element && !id.startsWith('starwave.')) {
        const section = element.closest('section[data-sw-section]');
        if (section) target.hash = section.dataset.swAnchor || section.dataset.swSection;
      }
    }
    return target.href;
  }
  document.querySelectorAll('[data-sw-language]').forEach(link => {
    const base = link.getAttribute('href');
    function refresh() { link.setAttribute('href', base); link.href = destination(link); }
    refresh();
    window.addEventListener('hashchange', refresh);
  });
});
