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
