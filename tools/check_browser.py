"""Browser regression checks for the built bilingual site (no GPU required)."""
import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
from urllib.parse import parse_qsl, quote, unquote, urlsplit
from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]
API_NAMES = ('scalar', 'vrz', 'vti', 'native_status', 'prepare_native')
USAGE_TARGETS = tuple(
    target
    for name in ('scalar', 'vrz', 'vti')
    for target in (name, f'{name}-function', f'{name}-details',
                   f'{name}-examples', f'{name}-notes')
) + ('usage', 'propagators', 'native-runtime', 'native-status', 'prepare-native',
     'native-notes', 'native-examples', 'other-exports')
TOC_TARGETS = ('scalar', 'vrz', 'vti', 'native-status', 'prepare-native', 'other-exports')
CHAPTERS = {
    'installation', 'wsl', 'quickstart', 'modeling/conventions',
    'modeling/scalar', 'modeling/vrz', 'modeling/vti', 'inversion/fwi',
    'inversion/dataparallel', 'inversion/inr', 'usage', 'faq',
    'release-notes', 'status',
}


def open_mobile_menu(page):
    if page.viewport_size['width'] <= 768 and not page.locator('.wy-nav-side').evaluate("e => e.classList.contains('shift')"):
        page.locator('[data-sw-mobile-menu]').click()
        page.wait_for_function("document.querySelector('[data-sw-mobile-menu]').getAttribute('aria-expanded') === 'true'")


def close_mobile_menu(page):
    if page.viewport_size['width'] <= 768 and page.locator('.wy-nav-side').evaluate("e => e.classList.contains('shift')"):
        page.locator('[data-sw-mobile-menu]').click()
        page.wait_for_function("document.querySelector('[data-sw-mobile-menu]').getAttribute('aria-expanded') === 'false'")


def check_navigation(page, locale, relative):
    assert page.locator('[data-sw-language]').count() == 2, relative
    nav = page.locator('.wy-menu-vertical')
    assert nav.locator('li ul').count() == 0, f'Nested sidebar: {relative}'
    assert nav.locator('.toctree-expand').count() == 0, f'Expandable sidebar: {relative}'
    links = nav.locator('a').evaluate_all('(items) => items.map(a => ({href:a.href, text:a.textContent.trim()}))')
    paths = {urlsplit(link['href']).path for link in links}
    expected = {f'/{locale}/{chapter}.html' for chapter in CHAPTERS}
    assert expected <= paths, f'Missing chapters on {relative}: {expected - paths}'
    assert not any('/api/' in path for path in paths), relative
    usage = [link for link in links if urlsplit(link['href']).path == f'/{locale}/usage.html']
    assert len(usage) == 1 and usage[0]['text'] == 'Usage', usage
    assert all(not urlsplit(link['href']).fragment for link in links), relative
    search = page.locator('#rtd-search-form input[name="q"]')
    assert search.count() == 1, relative
    assert search.get_attribute('placeholder') == 'Search docs', relative
    label = search.evaluate("e => e.getAttribute('aria-label') || Array.from(e.labels || []).map(l => l.textContent).join(' ')")
    assert label and (re.search(r'[\u4e00-\u9fff]', label) if locale == 'zh' else 'search' in label.lower()), (relative, label)
    if page.viewport_size['width'] > 768:
        assert search.is_visible(), relative
        box = search.bounding_box()
        sidebar = page.locator('.wy-nav-side').bounding_box()
        assert box['y'] < 400 and box['width'] >= 150 and box['height'] >= 32, box
        assert sidebar['x'] <= box['x'] < sidebar['x'] + sidebar['width'], box


def check_usage(page, locale, output):
    main = page.locator('[role="main"]')
    headings = main.locator('h1').evaluate_all("items => items.map(e => e.cloneNode(true)).map(e => {e.querySelectorAll('.headerlink').forEach(a => a.remove()); return e.textContent.trim();})")
    assert headings == ['Usage'], headings
    ids = main.locator('[id]').evaluate_all('(items) => items.map(e => e.id)')
    assert len(ids) == len(set(ids)), f'Duplicate Usage IDs: {locale}'
    for target in USAGE_TARGETS:
        assert ids.count(target) == 1, (locale, target)
    definitions = main.locator('dt.sig[id^="starwave."]').evaluate_all('(items) => items.map(e => e.id)')
    assert definitions == [f'starwave.{name}' for name in API_NAMES], definitions
    for name in API_NAMES:
        assert ids.count(f'starwave.{name}') == 1, (locale, name)
    toc = page.locator('.sw-page-toc')
    assert toc.count() == 1 and toc.is_visible(), locale
    assert toc.evaluate("e => ['Top', 'Right', 'Bottom', 'Left'].every(side => parseFloat(getComputedStyle(e)['border' + side + 'Width']) > 0)"), locale
    toc_links = toc.locator('a').evaluate_all('(items) => items.map(a => a.getAttribute("href"))')
    assert toc_links and all(link.startswith('#') for link in toc_links), toc_links
    assert [unquote(link[1:]) for link in toc_links] == list(TOC_TARGETS), toc_links
    assert all(unquote(link[1:]) in ids for link in toc_links), toc_links
    assert toc.evaluate("e => Boolean(e.compareDocumentPosition(document.getElementById('starwave.scalar')) & Node.DOCUMENT_POSITION_FOLLOWING)"), locale
    width = page.viewport_size['width']
    page.evaluate('window.scrollTo(0, 0)')
    page.screenshot(path=str(output / f'{locale}-usage-top-{width}.png'))
    # A focused viewport capture makes the actual parameter fields reviewable,
    # rather than shrinking the entire long Usage page into one image.
    fields = page.locator('[id="starwave.scalar"]').locator('..').locator('dl.field-list').first
    text = fields.inner_text()
    assert ('参数' if locale == 'zh' else 'Parameters') in text, locale
    assert ('必填' if locale == 'zh' else 'Required') in text, locale
    assert 'source_amplitudes' in text, locale
    fields.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.screenshot(path=str(output / f'{locale}-usage-scalar-fields-{width}.png'))
    # Exercise a real local TOC click, not only direct fragment navigation.
    toc.locator('a[href="#scalar"]').click()
    assert urlsplit(page.url).fragment == 'scalar'
    assert page.locator('[id="scalar"]').count() == 1



def check_heading_language_switches(page, base):
    for locale in ('en', 'zh'):
        other = 'zh' if locale == 'en' else 'en'
        page.goto(base + locale + '/usage.html')
        sections = page.locator('[role="main"] section[data-sw-anchor]').evaluate_all("""items => items.map(section => {
            const heading = Array.from(section.children).find(e => /^H[1-6]$/.test(e.tagName));
            return {id: section.id, stable: section.dataset.swAnchor,
                    permalink: heading?.querySelector('.headerlink')?.getAttribute('href')};
        })""")
        assert len(sections) == len(USAGE_TARGETS), (locale, sections)
        assert {section['stable'] for section in sections} == set(USAGE_TARGETS), (locale, sections)
        if locale == 'en':
            mappings = {section['id']: section['stable'] for section in sections}
            assert mappings['scalar-2d-scalar-acoustics'] == 'scalar', mappings
            assert mappings['sources-and-autograd'] == 'scalar-details', mappings
        for section in sections:
            assert section['permalink'] == '#' + section['id'], (locale, section)
            query = 'q=illumination&from=heading'
            page.goto(base + locale + '/usage.html?' + query + section['permalink'])
            assert page.locator(f'[id="{section["id"]}"]').count() == 1, (locale, section)
            open_mobile_menu(page)
            page.locator(f'[data-sw-language="{other}"]').click()
            destination = urlsplit(page.url)
            assert destination.path == f'/{other}/usage.html', (locale, section, page.url)
            assert unquote(destination.fragment) == section['stable'], (locale, section, page.url)
            assert destination.query == query, (locale, section, page.url)
            assert page.locator(f'[id="{section["stable"]}"]').count() == 1, (locale, section)

    # Ordinary chapters retain their positional, cross-language section aliases.
    page.goto(base + 'en/installation.html?from=chapter#runtime-requirements')
    open_mobile_menu(page)
    page.locator('[data-sw-language="zh"]').click()
    assert urlsplit(page.url).path == '/zh/installation.html'
    assert urlsplit(page.url).fragment == 'sw-section-1'
    assert urlsplit(page.url).query == 'from=chapter'
    assert page.locator('#sw-section-1').count() == 1
    open_mobile_menu(page)
    page.locator('[data-sw-language="en"]').click()
    assert urlsplit(page.url).path == '/en/installation.html'
    assert urlsplit(page.url).fragment == 'sw-section-1'


def check_redirects(page, base):
    redirects = json.loads((PROJECT / 'tools' / 'api_redirects.json').read_text())
    for source_locale in ('root', 'zh', 'en'):
        locale = 'zh' if source_locale == 'root' else source_locale
        prefix = '' if source_locale == 'root' else source_locale + '/'
        for relative, spec in redirects[source_locale].items():
            source_path = prefix + relative
            fragments = {'': spec['default']} | spec['fragments']
            for old, target in fragments.items():
                query = 'q=source%20amplitudes&from=legacy'
                source = base + source_path + '?' + query + ('#' + quote(old) if old else '')
                documents = []

                def record(request):
                    if request.is_navigation_request() and request.frame == page.main_frame:
                        documents.append(urlsplit(request.url).path)

                page.on('request', record)
                try:
                    page.goto(source, wait_until='load')
                    page.wait_for_url(lambda url: urlsplit(str(url)).path == f'/{locale}/usage.html', wait_until='load')
                    parsed = urlsplit(page.url)
                    # Sphinx may normalize a query space from %20 to + on load.
                    assert parse_qsl(parsed.query, keep_blank_values=True) == parse_qsl(query, keep_blank_values=True), (source, page.url)
                    assert unquote(parsed.fragment) == target, (source, page.url, target)
                    assert documents == ['/' + source_path, f'/{locale}/usage.html'], (source, documents)
                    assert page.locator(f'[id="{target}"]').count() == 1, (source, target)
                finally:
                    page.remove_listener('request', record)


def check_search(page, base):
    for locale in ('en', 'zh'):
        page.goto(base + locale + '/index.html')
        open_mobile_menu(page)
        search = page.locator('#rtd-search-form input[name="q"]')
        assert search.is_visible()
        search.fill('illumination')
        search.press('Enter')
        page.wait_for_url(f'**/{locale}/search.html?*')
        assert 'q=illumination' in urlsplit(page.url).query
        page.wait_for_function('document.querySelectorAll("#search-results li").length > 0')
        expected_status = '搜索完成' if locale == 'zh' else 'Search finished'
        page.wait_for_function("expected => document.querySelector('.search-summary')?.textContent.includes(expected)", arg=expected_status)
        links = page.locator('#search-results li a').evaluate_all('(items) => items.map(a => a.href)')
        assert links and all(f'/{locale}/' in urlsplit(link).path and '/api/' not in urlsplit(link).path for link in links), links
        index = page.evaluate('Search._index')
        assert 'usage' in index['docnames'] and not any(name.startswith('api/') for name in index['docnames']), index['docnames']
        objects = {item[4]: item[0] for item in index['objects'].get('starwave', [])}
        assert set(API_NAMES) <= objects.keys(), objects
        assert all(index['docnames'][objects[name]] == 'usage' for name in API_NAMES), objects
        other = 'zh' if locale == 'en' else 'en'
        open_mobile_menu(page)
        page.locator(f'[data-sw-language="{other}"]').click()
        assert urlsplit(page.url).path == f'/{other}/search.html'
        assert 'q=illumination' in urlsplit(page.url).query
        page.goto(base + locale + '/search.html?q=zzzznosuchstarwaveresult')
        expected_status = '搜索未找到匹配页面' if locale == 'zh' else 'Your search did not match any documents'
        page.wait_for_function("expected => document.querySelector('.search-summary')?.textContent.includes(expected)", arg=expected_status)


def check(root, browser_path=None):
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}/'
    output = PROJECT / '_build' / 'qa'
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    try:
        with sync_playwright() as playwright:
            options = {'executable_path': browser_path} if browser_path else {}
            browser = playwright.chromium.launch(**options)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            for size in ({'width': 1440, 'height': 1000}, {'width': 390, 'height': 844}):
                page.set_viewport_size(size)
                for locale in ('zh', 'en'):
                    for path in sorted((root / locale).rglob('*.html')):
                        if path.relative_to(root / locale).parts[0] == 'api':
                            continue  # Compatibility pages are tested separately.
                        relative = path.relative_to(root).as_posix()
                        page.goto(base + relative, wait_until='networkidle')
                        check_navigation(page, locale, relative)
                        if not page.evaluate('document.documentElement.scrollWidth <= innerWidth'):
                            page.screenshot(path=str(output / f'overflow-{locale}-{path.stem}-{size["width"]}.png'), full_page=True)
                            wide = page.evaluate("""Array.from(document.querySelectorAll('body *')).filter(e => e.getBoundingClientRect().right > innerWidth).map(e => ({tag:e.tagName, classes:e.className, text:e.textContent.slice(0,120)})).slice(-12)""")
                            raise AssertionError(f'Horizontal overflow: {relative} at {size}; {wide}')
                    page.goto(base + locale + '/usage.html', wait_until='networkidle')
                    check_usage(page, locale, output)
                check_search(page, base)
            page.goto(base + 'zh/usage.html')
            open_mobile_menu(page)
            page.screenshot(path=str(output / 'zh-mobile-navigation.png'))
            close_mobile_menu(page)
            open_mobile_menu(page)
            close_mobile_menu(page)
            assert '开始使用' in page.locator('.wy-nav-side').inner_text()
            page.goto(base + 'zh/api/scalar.html?q=illumination#id4')
            page.wait_for_url('**/zh/usage.html?q=illumination#scalar-examples')
            open_mobile_menu(page)
            page.locator('[data-sw-language="en"]').click()
            assert urlsplit(page.url).path == '/en/usage.html'
            assert urlsplit(page.url).fragment == 'scalar-examples'
            assert urlsplit(page.url).query == 'q=illumination'
            open_mobile_menu(page)
            page.locator('[data-sw-language="zh"]').click()
            assert urlsplit(page.url).path == '/zh/usage.html'
            assert urlsplit(page.url).fragment == 'scalar-examples'
            page.go_back()
            assert urlsplit(page.url).path == '/en/usage.html'
            page.go_forward()
            assert urlsplit(page.url).path == '/zh/usage.html'
            for name in API_NAMES:
                page.goto(base + f'zh/usage.html#starwave.{name}')
                open_mobile_menu(page)
                page.locator('[data-sw-language="en"]').click()
                assert urlsplit(page.url).path == '/en/usage.html'
                assert urlsplit(page.url).fragment == f'starwave.{name}'
                assert page.locator(f'[id="starwave.{name}"]').count() == 1
            check_heading_language_switches(page, base)
            check_redirects(page, base)
            page.goto(base + 'en/index.html')
            page.keyboard.press('Tab')
            assert page.locator('.skip-link').evaluate('(e) => e === document.activeElement')
            page.keyboard.press('Enter')
            assert page.locator('#main-content').evaluate('(e) => e === document.activeElement')
            context = browser.new_context(java_script_enabled=False, viewport={'width': 1440, 'height': 1000})
            plain = context.new_page()
            plain.goto(base + 'en/usage.html')
            assert plain.locator('.sw-page-toc').is_visible()
            plain.locator('.sw-page-toc a[href="#scalar"]').click()
            assert urlsplit(plain.url).fragment == 'scalar'
            plain.locator('[data-sw-language="zh"]').click()
            assert urlsplit(plain.url).path == '/zh/usage.html'
            for prefix in ('', 'zh/', 'en/'):
                for name in ('index', 'scalar', 'vrz', 'vti'):
                    plain.goto(base + f'{prefix}api/{name}.html')
                    plain.locator('[data-sw-language="en"]').click()
                    assert urlsplit(plain.url).path == '/en/usage.html'
                    assert urlsplit(plain.url).fragment == ('usage' if name == 'index' else name)
            context.close()
            browser.close()
        assert not errors, errors
        print('PASS: flat desktop/mobile navigation, Usage TOC and API fields, search, direct legacy routes, language anchors, history, keyboard and no-JS fallback.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / '_build' / 'html')
    parser.add_argument('--browser-path')
    args = parser.parse_args()
    check(args.root.resolve(), args.browser_path)
