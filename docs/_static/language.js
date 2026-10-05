/* Same-page locale switching, including nested API sections and search queries. */
document.addEventListener('DOMContentLoaded', () => {
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
        if (section) target.hash = section.dataset.swSection;
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
