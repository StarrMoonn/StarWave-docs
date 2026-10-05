"""Browser regression checks for the built bilingual site (no GPU required)."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]


def open_mobile_menu(page):
    if page.viewport_size['width'] <= 768 and not page.locator('.wy-nav-side').evaluate("e => e.classList.contains('shift')"):
        page.locator('[data-sw-mobile-menu]').click()
        page.wait_for_function("document.querySelector('[data-sw-mobile-menu]').getAttribute('aria-expanded') === 'true'")


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
                        relative = path.relative_to(root).as_posix()
                        page.goto(base + relative, wait_until='networkidle')
                        assert page.locator('[data-sw-language]').count() == 2, relative
                        assert page.locator('.wy-nav-side .caption-text').filter(has_text='Usage / API').count() == 1, relative
                        assert page.locator('.wy-nav-side a').filter(has_text='starwave.native_status').count() == 1, relative
                        assert page.locator('.wy-nav-side a').filter(has_text='starwave.prepare_native').count() == 1, relative
                        if not page.evaluate('document.documentElement.scrollWidth <= innerWidth'):
                            page.screenshot(path=str(output / f'overflow-{locale}-{path.stem}-{size["width"]}.png'), full_page=True)
                            wide = page.evaluate("""Array.from(document.querySelectorAll('body *')).filter(e => e.getBoundingClientRect().right > innerWidth).map(e => ({tag:e.tagName, classes:e.className, text:e.textContent.slice(0,120)})).slice(-12)""")
                            raise AssertionError(f'Horizontal overflow: {relative} at {size}; {wide}')
                    page.goto(base + locale + '/api/scalar.html')
                    page.screenshot(path=str(output / f'{locale}-scalar-{size["width"]}.png'), full_page=True)
            page.goto(base + 'zh/api/scalar.html')
            open_mobile_menu(page)
            page.screenshot(path=str(output / 'zh-mobile-navigation.png'))
            page.locator('[data-sw-mobile-menu]').click()
            page.wait_for_function("document.querySelector('[data-sw-mobile-menu]').getAttribute('aria-expanded') === 'false'")
            open_mobile_menu(page)
            page.locator('[data-sw-mobile-menu]').click()
            assert '参数' in page.locator('[role="main"]').inner_text()
            assert '必填' in page.locator('[role="main"]').inner_text()
            assert 'source_amplitudes' in page.locator('[role="main"]').inner_text()
            assert '开始使用' in page.locator('.wy-nav-side').inner_text()
            page.goto(base + 'zh/api/scalar.html#id4')
            open_mobile_menu(page)
            page.locator('[data-sw-language="en"]').click()
            assert urlsplit(page.url).path == '/en/api/scalar.html'
            assert 'Parameters' in page.locator('[role="main"]').inner_text()
            assert 'Required' in page.locator('[role="main"]').inner_text()
            assert urlsplit(page.url).fragment == 'sw-section-3'
            assert page.locator('#sw-section-3').count() == 1
            open_mobile_menu(page)
            page.locator('[data-sw-language="zh"]').click()
            assert urlsplit(page.url).path == '/zh/api/scalar.html'
            assert urlsplit(page.url).fragment == 'sw-section-3'
            page.go_back()
            assert '/en/api/scalar.html' in page.url
            page.go_forward()
            assert '/zh/api/scalar.html' in page.url
            for name in ('scalar', 'vrz', 'vti'):
                page.goto(base + f'api/{name}.html#starwave.{name}')
                page.wait_for_url(f'**/zh/api/{name}.html#starwave.{name}')
                open_mobile_menu(page)
                page.locator('[data-sw-language="en"]').click()
                assert page.locator(f'[id="starwave.{name}"]').count() == 1
                assert urlsplit(page.url).fragment == f'starwave.{name}'
            page.goto(base + 'api/scalar.html#id4')
            page.wait_for_url('**/zh/api/scalar.html#sw-section-3')
            for locale in ('en', 'zh'):
                page.goto(base + locale + '/search.html?q=illumination')
                page.wait_for_function('document.querySelectorAll("#search-results li").length > 0')
                expected_status = '搜索完成' if locale == 'zh' else 'Search finished'
                page.wait_for_function("expected => document.querySelector('.search-summary')?.textContent.includes(expected)", arg=expected_status)
                links = page.locator('#search-results li a').evaluate_all('(items) => items.map(a => a.href)')
                assert links and all(f'/{locale}/' in urlsplit(link).path for link in links), links
                other = 'zh' if locale == 'en' else 'en'
                open_mobile_menu(page)
                page.locator(f'[data-sw-language="{other}"]').click()
                assert urlsplit(page.url).query == 'q=illumination'
            for locale in ('en', 'zh'):
                page.goto(base + locale + '/search.html?q=zzzznosuchstarwaveresult')
                expected_status = '搜索未找到匹配页面' if locale == 'zh' else 'Your search did not match any documents'
                page.wait_for_function("expected => document.querySelector('.search-summary')?.textContent.includes(expected)", arg=expected_status)
            page.goto(base + 'en/index.html')
            page.keyboard.press('Tab')
            assert page.locator('.skip-link').evaluate('(e) => e === document.activeElement')
            page.keyboard.press('Enter')
            assert page.locator('#main-content').evaluate('(e) => e === document.activeElement')
            context = browser.new_context(java_script_enabled=False)
            plain = context.new_page()
            plain.goto(base + 'en/api/vti.html')
            plain.locator('[data-sw-language="zh"]').click()
            assert urlsplit(plain.url).path == '/zh/api/vti.html'
            plain.goto(base + 'api/scalar.html')
            plain.get_by_role('link', name='Continue in English').click()
            assert urlsplit(plain.url).path == '/en/api/scalar.html'
            context.close()
            browser.close()
        assert not errors, errors
        print('PASS: desktop/mobile pages, language and API anchors, legacy routes, search, history, keyboard and no-JS fallback.')
    finally:
        server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / '_build' / 'html')
    parser.add_argument('--browser-path')
    args = parser.parse_args()
    check(args.root.resolve(), args.browser_path)
