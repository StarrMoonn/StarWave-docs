"""Targeted browser QA for the bilingual 2-D free-surface chapter.

This checks publication behavior, not numerical or physical acceptance.
Uses an ephemeral loopback server and the Chrome installed by documentation CI.
"""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

from playwright.sync_api import sync_playwright

from check_browser import close_mobile_menu, open_mobile_menu, wait_for_native_condition
from check_free_surface_examples import CASES, VALIDATIONS

PROJECT = Path(__file__).resolve().parents[1]
CHAPTER = 'modeling/free-surface.html'
FIGURES = set(CASES + VALIDATIONS + ('born_background',))


def check_layout(page, locale, width):
    """Allow table-local scrolling, never document-wide horizontal overflow."""
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), (
        'Document overflow', locale, width)
    assert page.evaluate('document.body.scrollWidth <= innerWidth + 1'), (
        'Body overflow', locale, width)
    tables = page.locator('[role="main"] table')
    assert tables.count() == 3, ('Missing comparison/refinement/verification table', locale)
    geometry = tables.evaluate_all("""tables => tables.map(table => {
        const container = table.closest('.wy-table-responsive') || table.parentElement;
        const main = table.closest('[role="main"]');
        const box = container.getBoundingClientRect();
        const tableBox = table.getBoundingClientRect();
        const mainBox = main.getBoundingClientRect();
        const overflow = getComputedStyle(container).overflowX;
        const bounded = box.left >= mainBox.left - 1 && box.right <= mainBox.right + 1
            && box.left >= -1 && box.right <= innerWidth + 1;
        const fits = tableBox.left >= box.left - 1 && tableBox.right <= box.right + 1;
        const scrolls = ['auto', 'scroll'].includes(overflow)
            && container.scrollWidth >= container.clientWidth;
        return {bounded, fits, scrolls, overflow,
            containerWidth: box.width, tableWidth: tableBox.width};
    })""")
    assert all(item['bounded'] and (item['fits'] or item['scrolls']) for item in geometry), (
        'Table escapes its container', locale, width, geometry)
    return geometry


def check_images(page, locale, width):
    images = page.locator('[role="main"] figure.sw-free-surface-figure img')
    assert images.count() == 12, ('Expected twelve numerical figures', locale, width)
    names = images.evaluate_all('images => images.map(img => new URL(img.src).pathname.split("/").pop())')
    assert set(names) == {name + '.png' for name in FIGURES}, (locale, names)
    for image in images.all():
        image.scroll_into_view_if_needed()
        wait_for_native_condition(image,
            'img => img.complete && img.naturalWidth > 0 && img.naturalHeight > 0',
            f'Free-surface PNG at {locale}/{width}')
        assert image.evaluate("""img => {
            const box = img.getBoundingClientRect();
            const main = img.closest('[role="main"]').getBoundingClientRect();
            return box.width > 0 && box.left >= main.left - 1 && box.right <= main.right + 1;
        }"""), ('Image escapes article', locale, width, image.get_attribute('src'))
        assert image.get_attribute('alt'), ('Missing figure alternative', locale)


def check_downloads(page, locale, root):
    """Exercise the exact user-visible data links and compare served bytes."""
    links = page.locator('[role="main"] a[href]').evaluate_all(r"""items =>
        items.map(a => a.href).filter(href => {
            const path = new URL(href).pathname;
            return path.includes('/_static/free-surface/') && /\.(npz|json)$/.test(path);
        })""")
    expected = {name + '.' + extension for name in CASES + VALIDATIONS
                for extension in ('npz', 'json')} | {'manifest.json'}
    assert {urlsplit(link).path.rsplit('/', 1)[-1] for link in links} == expected, (
        'Missing data-download link', locale)
    for link in sorted(set(links)):
        parsed = urlsplit(link)
        assert parsed.path.startswith(f'/{locale}/_static/free-surface/'), (locale, link)
        response = page.request.get(link)
        assert response.ok, ('Data download failed', locale, link, response.status)
        local = root / parsed.path.lstrip('/')
        assert hashlib.sha256(response.body()).digest() == hashlib.sha256(local.read_bytes()).digest(), (
            'Served data differs from publication', locale, link)
        response.dispose()
    return len(set(links))


def check_language_switch(page, base, locale):
    """The existing language script maps localized headings to stable sections."""
    other = 'en' if locale == 'zh' else 'zh'
    page.goto(base + locale + '/' + CHAPTER + '?from=free-surface-qa#free-surface-mirror')
    mirror = page.locator('[id="free-surface-mirror"]')
    assert mirror.count() == 1
    target = mirror.evaluate("""element => {
        const section = element.closest('section[data-sw-section]');
        return section.dataset.swAnchor || section.dataset.swSection;
    }""")
    open_mobile_menu(page)
    page.locator(f'[data-sw-language="{other}"]').click()
    page.wait_for_url(base + other + '/' + CHAPTER + '?from=free-surface-qa#' + target)
    assert urlsplit(page.url).query == 'from=free-surface-qa'
    assert unquote(urlsplit(page.url).fragment) == target
    anchor = page.locator(f'[id="{target}"]')
    assert anchor.count() == 1
    assert anchor.evaluate("e => !!e.closest('section').querySelector('[id=\"free-surface-mirror\"]')"), (
        'Language switch reached the wrong section', locale, target)
    close_mobile_menu(page)
    page.go_back()
    page.wait_for_url(base + locale + '/' + CHAPTER + '?from=free-surface-qa#free-surface-mirror')
    page.go_forward()
    page.wait_for_url(base + other + '/' + CHAPTER + '?from=free-surface-qa#' + target)
    return target


def check(root, browser_path=None, browser_channel='chrome'):
    assert (root / 'zh' / CHAPTER).is_file() and (root / 'en' / CHAPTER).is_file()
    output = PROJECT / '_build/qa'
    output.mkdir(parents=True, exist_ok=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}/'
    report = {'scope': 'Publication UI only; not physical acceptance', 'cases': []}
    errors = []
    try:
        with sync_playwright() as playwright:
            options = {'executable_path': browser_path} if browser_path else {'channel': browser_channel}
            browser = playwright.chromium.launch(**options)
            try:
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                page.on('pageerror', lambda error: errors.append(str(error)))
                for width in (1440, 390):
                    page.set_viewport_size({'width': width, 'height': 1000 if width == 1440 else 844})
                    for locale in ('zh', 'en'):
                        try:
                            page.goto(base + locale + '/index.html', wait_until='networkidle')
                            open_mobile_menu(page)
                            label = '自由表面' if locale == 'zh' else 'Free surface'
                            link = page.locator('.wy-menu-vertical a').filter(has_text=label)
                            assert link.count() == 1, ('Sidebar chapter is not unique', locale)
                            assert link.inner_text().strip() == label
                            link.click()
                            page.wait_for_url(base + locale + '/' + CHAPTER)
                            close_mobile_menu(page)
                            assert page.locator('[role="main"] h1').inner_text().startswith(label)
                            check_images(page, locale, width)
                            geometry = check_layout(page, locale, width)
                            downloads = check_downloads(page, locale, root) if width == 1440 else None
                            page.evaluate('window.scrollTo(0, 0)')
                            page.wait_for_function('window.scrollY === 0')
                            page.screenshot(path=str(output / f'{locale}-free-surface-top-{width}.png'))
                            mirror = page.locator('figure[data-validation="scalar_mirror"]')
                            mirror.scroll_into_view_if_needed()
                            mirror.screenshot(path=str(output / f'{locale}-free-surface-mirror-{width}.png'))
                            target = check_language_switch(page, base, locale)
                            report['cases'].append({'locale': locale, 'width': width, 'figures_loaded': 12,
                                'download_links_checked': downloads, 'table_geometry': geometry,
                                'language_target': target, 'status': 'passed'})
                        except Exception:
                            page.screenshot(path=str(output / f'{locale}-free-surface-failure-{width}.png'))
                            raise
                assert not errors, ('Browser page errors', errors)
            finally:
                browser.close()
        report['status'] = 'passed'
        print('PASS: free-surface bilingual navigation, twelve PNGs, raw downloads, desktop/mobile layout, same-section language switching and history.')
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        (output / 'free-surface-browser-qa.json').write_text(json.dumps(report, indent=2) + '\n')
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / '_build/html')
    parser.add_argument('--browser-path')
    parser.add_argument('--browser-channel', choices=('chromium', 'chrome'), default='chrome')
    args = parser.parse_args()
    check(args.root.resolve(), args.browser_path, args.browser_channel)
