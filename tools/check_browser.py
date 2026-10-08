"""Browser regression checks for the built bilingual site (no GPU required)."""
import argparse
import hashlib
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
from urllib.parse import parse_qsl, quote, unquote, urlsplit
from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]
API_NAMES = ('scalar', 'vrz', 'vti', 'elastic',
             'common.vpvsrho_to_lambmubuoyancy', 'common.lambmubuoyancy_to_vpvsrho',
             'native_status', 'prepare_native', 'prepare_elastic')
ELASTIC_SECTIONS = ('elastic', 'elastic-axes', 'elastic-returns', 'elastic-memory',
                    'elastic-examples', 'elastic-notes', 'elastic-conversions')
USAGE_TARGETS = tuple(
    target
    for name in ('scalar', 'vrz', 'vti')
    for target in (name, f'{name}-function', f'{name}-details',
                   f'{name}-examples', f'{name}-notes')
) + ELASTIC_SECTIONS + ('elastic-function', 'usage', 'propagators', 'native-runtime',
                       'native-status', 'prepare-native', 'native-notes',
                       'native-examples', 'prepare-elastic', 'other-exports')
TOC_TARGETS = ('scalar', 'vrz', 'vti', 'elastic', 'elastic-conversions',
               'native-status', 'prepare-native', 'prepare-elastic', 'other-exports')
USAGE_SECTIONS = tuple(target for target in USAGE_TARGETS if not target.endswith('-function'))
CHAPTERS = {
    'about', 'presentation', 'installation', 'docker', 'wsl', 'quickstart', 'modeling/conventions',
    'modeling/scalar', 'modeling/gradient', 'modeling/acquisition', 'modeling/wave-propagation', 'inversion/fwi',
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
    assert not any('/api/' in path or path.endswith(('/modeling/vrz.html', '/modeling/vti.html')) for path in paths), relative
    wave_links = [link for link in links if urlsplit(link['href']).path == f'/{locale}/modeling/wave-propagation.html']
    assert len(wave_links) == 1 and wave_links[0]['text'] == ('波传播' if locale == 'zh' else 'Wave propagation'), (relative, wave_links)
    captions = nav.locator('.caption-text').all_text_contents()
    assert ('正演模拟' if locale == 'zh' else 'Forward Modeling') in captions, (relative, captions)
    assert next(link['text'] for link in links if urlsplit(link['href']).path == f'/{locale}/wsl.html') == 'WSL'
    usage = [link for link in links if urlsplit(link['href']).path == f'/{locale}/usage.html']
    assert len(usage) == 1 and usage[0]['text'] == 'Usage', usage
    assert all(not urlsplit(link['href']).fragment for link in links), relative
    # Chapter labels stay readable on one line at the existing sidebar size.
    assert nav.locator('a').evaluate_all("""items => items.every(a => {
        const range = document.createRange(); range.selectNodeContents(a);
        const rows = Array.from(range.getClientRects()).filter(r => r.width && r.height);
        return new Set(rows.map(r => Math.round(r.top))).size <= 1 && a.scrollWidth <= a.clientWidth;
    })"""), f'Wrapped or clipped sidebar label: {relative}'
    search = page.locator('#rtd-search-form input[name="q"]')
    assert search.count() == 1, relative
    assert search.get_attribute('placeholder') == 'Search docs', relative
    submit = page.locator('#rtd-search-form button[type="submit"]')
    assert submit.count() == 1 and submit.get_attribute('aria-label'), relative
    input_box, button_box = search.bounding_box(), submit.bounding_box()
    if input_box and button_box:
        assert input_box['x'] <= button_box['x'] < input_box['x'] + input_box['width'] / 2, relative
        assert button_box['width'] >= 40 and button_box['height'] >= 40, relative
        assert abs(button_box['y'] + button_box['height'] / 2 - input_box['y'] - input_box['height'] / 2) <= 1, (relative, input_box, button_box)
        svg_box = submit.locator('svg').bounding_box()
        assert svg_box and abs(svg_box['y'] + svg_box['height'] / 2 - input_box['y'] - input_box['height'] / 2) <= 1, (relative, svg_box)
        assert button_box['x'] + button_box['width'] <= input_box['x'] + input_box['width'], relative
    label = search.evaluate("e => e.getAttribute('aria-label') || Array.from(e.labels || []).map(l => l.textContent).join(' ')")
    assert label and (re.search(r'[\u4e00-\u9fff]', label) if locale == 'zh' else 'search' in label.lower()), (relative, label)
    if page.viewport_size['width'] > 768:
        assert search.is_visible(), relative
        box = search.bounding_box()
        sidebar = page.locator('.wy-nav-side').bounding_box()
        assert 0 <= box['y'] < 400 and box['y'] + box['height'] <= page.viewport_size['height'], box
        assert box['width'] >= 150 and box['height'] >= 32, box
        assert sidebar['x'] <= box['x'] < sidebar['x'] + sidebar['width'], box


def check_usage(page, locale, output):
    main = page.locator('[role="main"]')
    headings = main.locator('h1').evaluate_all("items => items.map(e => e.cloneNode(true)).map(e => {e.querySelectorAll('.headerlink').forEach(a => a.remove()); return e.textContent.trim();})")
    assert headings == ['Usage'], headings
    ids = main.locator('[id]').evaluate_all('(items) => items.map(e => e.id)')
    assert len(ids) == len(set(ids)), f'Duplicate Usage IDs: {locale}'
    for target in USAGE_TARGETS:
        assert ids.count(target) == 1, (locale, target)
    page.evaluate('document.fonts.ready')
    title = main.locator('#usage > h1').evaluate("e => ({size:parseFloat(getComputedStyle(e).fontSize),font:getComputedStyle(e).fontFamily,weight:getComputedStyle(e).fontWeight})")
    assert abs(title['size'] - 40.8) < .01 and 'Georgia' in title['font'] and title['weight'] == '400', title
    for name, label in (('scalar', 'Scalar Function'), ('vrz', 'VRZ Function'),
                        ('vti', 'VTI Function'), ('elastic', 'Elastic Function')):
        heading = main.locator(f'#{name} > h2')
        assert heading.evaluate("e => e.firstChild.textContent.trim()") == label
        assert abs(heading.evaluate('e => parseFloat(getComputedStyle(e).fontSize)') - 30.6) < .01
        assert 'Georgia' in heading.evaluate('e => getComputedStyle(e).fontFamily')
    assert not main.locator('h3').evaluate_all("items => items.some(e => ['Function', '函数'].includes(e.firstChild.textContent.trim()))")
    overview = main.locator('#propagators')
    assert overview.locator('code.xref').all_text_contents() == ['starwave.scalar()', 'starwave.vrz()', 'starwave.vti()', 'starwave.elastic()']
    assert overview.locator(':scope > p code.literal').all_text_contents() == ['B', 'S', 'R', 'T', 'D']
    overview_rows = overview.locator('p').evaluate_all("""items => items.map(p => {
        const prose = getComputedStyle(p);
        return {size: prose.fontSize, line: prose.lineHeight,
                codes: Array.from(p.querySelectorAll('a, code, code span')).map(e => {
                    const s = getComputedStyle(e);
                    return {text:e.textContent, size:s.fontSize, line:s.lineHeight, vertical:s.verticalAlign};
                })};
    })""")
    assert len(overview_rows) == 5
    assert len({row['line'] for row in overview_rows}) == 1, (locale, overview_rows)
    assert all(row['size'] == '17px' and all(code['size'] == row['size'] and code['line'] == row['line'] and code['vertical'] == 'baseline' for code in row['codes']) for row in overview_rows), (locale, overview_rows)
    if locale == 'en':
        dimensions = overview.locator('code.literal').filter(has_text=re.compile(r'^2D(?:/3D)?$'))
        assert dimensions.all_text_contents() == ['2D', '2D', '2D/3D', '2D/3D']
        # Inspect the rendered font metrics: dimension numerals must have lining
        # figures rather than a descending old-style 3 beside the uppercase D.
        metrics = dimensions.first.evaluate("""e => {
            const s = getComputedStyle(e), ctx = document.createElement('canvas').getContext('2d');
            ctx.font = `${s.fontStyle} ${s.fontWeight} ${s.fontSize} ${s.fontFamily}`;
            return {font:s.fontFamily, variant:s.fontVariantNumeric,
                    glyphs:Array.from('23D').map(c => { const m=ctx.measureText(c);
                        return {ascent:m.actualBoundingBoxAscent, descent:m.actualBoundingBoxDescent}; })};
        }""")
        assert 'monospace' in metrics['font'] and metrics['variant'] == 'lining-nums', metrics
        assert max(g['ascent'] for g in metrics['glyphs']) - min(g['ascent'] for g in metrics['glyphs']) <= 2, metrics
        assert all(g['descent'] <= 1 for g in metrics['glyphs']), metrics
    for name, count in (('scalar', 16), ('vrz', 18), ('vti', 20), ('elastic', 62),
                        ('common.vpvsrho_to_lambmubuoyancy', 4),
                        ('common.lambmubuoyancy_to_vpvsrho', 4),
                        ('prepare_native', 1), ('prepare_elastic', 1)):
        parameters = main.locator(f'dt[id="starwave.{name}"]').locator('..').locator('dl.field-list > dd').first.locator('p')
        assert parameters.count() == count, (locale, name, parameters.count())
        rows = parameters.evaluate_all("""items => items.map(p => ({
            name: p.querySelector('strong')?.textContent,
            type: p.querySelector('code.literal')?.textContent,
            sizes: [p, ...p.querySelectorAll('strong, code, span, em')].map(e => parseFloat(getComputedStyle(e).fontSize))
        }))""")
        assert all(row['name'] and row['type'] and all(size == 17 for size in row['sizes']) for row in rows), (locale, name, rows)
    for signature in main.locator('dt.sig[id^="starwave."]').all():
        style = signature.evaluate("e => { const s = getComputedStyle(e); return {font:s.fontFamily,size:parseFloat(s.fontSize),line:parseFloat(s.lineHeight),border:s.borderTopWidth,bg:s.backgroundColor}; }")
        assert 'Consolas' in style['font'] and 'monospace' in style['font'] and style['size'] == 17, style
        assert style['border'] == '0px' and style['bg'] == 'rgba(0, 0, 0, 0)', style
        assert abs(signature.locator('.sig-name').evaluate('e => parseFloat(getComputedStyle(e).fontSize)') - 18.7) < .1
    assert main.locator('code.literal').first.evaluate('e => getComputedStyle(e).borderTopWidth') == '0px'
    if locale == 'en':
        assert 'Georgia' in main.locator('#usage').evaluate('e => getComputedStyle(e).fontFamily')
    else:
        assert page.evaluate('document.fonts.check(\'400 17px "StarWave Home Sans"\', "参数")')
    definitions = main.locator('dt.sig[id^="starwave."]').evaluate_all('(items) => items.map(e => e.id)')
    assert definitions == [f'starwave.{name}' for name in API_NAMES], definitions
    for name in API_NAMES:
        assert ids.count(f'starwave.{name}') == 1, (locale, name)
    for name in ELASTIC_SECTIONS + ('prepare-elastic',):
        section = main.locator(f'section[id="{name}"]')
        assert section.get_attribute('data-sw-anchor') == name, (locale, name)
        assert section.locator(':scope > h2, :scope > h3').count() == 1, (locale, name)
    # The long native signature must wrap within the article, including on the
    # narrowest phone. The conversions remain independent, linkable API entries.
    for name in ('elastic', 'common.vpvsrho_to_lambmubuoyancy',
                 'common.lambmubuoyancy_to_vpvsrho', 'prepare_elastic'):
        signature = main.locator(f'dt[id="starwave.{name}"]')
        assert signature.evaluate('e => e.scrollWidth <= e.clientWidth + 1'), (locale, name, page.viewport_size)
    toc = page.locator('.sw-page-toc')
    assert toc.count() == 1 and toc.is_visible(), locale
    assert toc.evaluate("e => ['Top', 'Right', 'Bottom', 'Left'].every(side => parseFloat(getComputedStyle(e)['border' + side + 'Width']) > 0)"), locale
    toc_links = toc.locator('a').evaluate_all('(items) => items.map(a => a.getAttribute("href"))')
    assert toc_links and all(link.startswith('#') for link in toc_links), toc_links
    assert [unquote(link[1:]) for link in toc_links] == list(TOC_TARGETS), toc_links
    assert toc.locator('a').all_text_contents()[:4] == ['Scalar Function', 'VRZ Function', 'VTI Function', 'Elastic Function']
    assert all(unquote(link[1:]) in ids for link in toc_links), toc_links
    assert toc.evaluate("e => Boolean(e.compareDocumentPosition(document.getElementById('starwave.scalar')) & Node.DOCUMENT_POSITION_FOLLOWING)"), locale
    width = page.viewport_size['width']
    page.evaluate('window.scrollTo(0, 0)')
    page.screenshot(path=str(output / f'{locale}-usage-top-{width}.png'))
    overview.locator('h2').evaluate("e => e.scrollIntoView({block: 'start'})")
    page.screenshot(path=str(output / f'{locale}-usage-overview-{width}.png'))
    for name in ('scalar', 'vrz', 'vti', 'elastic'):
        main.locator(f'#{name} > h2').evaluate("e => e.scrollIntoView({block: 'start'})")
        page.screenshot(path=str(output / f'{locale}-usage-{name}-heading-{width}.png'))
    main.locator('[id="starwave.scalar"]').evaluate("e => e.scrollIntoView({block: 'start'})")
    page.screenshot(path=str(output / f'{locale}-usage-scalar-signature-{width}.png'))
    # A focused viewport capture makes the actual parameter fields reviewable,
    # rather than shrinking the entire long Usage page into one image.
    fields = page.locator('[id="starwave.scalar"]').locator('..').locator('dl.field-list').first
    text = fields.inner_text()
    assert ('参数' if locale == 'zh' else 'Parameters') in text, locale
    assert ('必填' if locale == 'zh' else 'Required') in text, locale
    assert 'source_amplitudes' in text, locale
    fields.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.screenshot(path=str(output / f'{locale}-usage-scalar-fields-{width}.png'))
    for name in ('elastic', 'common.vpvsrho_to_lambmubuoyancy',
                 'common.lambmubuoyancy_to_vpvsrho', 'prepare_elastic'):
        signature = main.locator(f'[id="starwave.{name}"]')
        signature.evaluate("e => e.scrollIntoView({block: 'start'})")
        page.screenshot(path=str(output / f'{locale}-usage-{name}-signature-{width}.png'))
        fields = signature.locator('..').locator('dl.field-list').first
        fields.evaluate("e => e.scrollIntoView({block: 'start'})")
        page.screenshot(path=str(output / f'{locale}-usage-{name}-fields-{width}.png'))
    # Exercise a real local TOC click, not only direct fragment navigation.
    toc.locator('a[href="#scalar"]').click()
    assert urlsplit(page.url).fragment == 'scalar'
    assert page.locator('[id="scalar"]').count() == 1
    for target in ('elastic', 'elastic-conversions', 'prepare-elastic'):
        toc.locator(f'a[href="#{target}"]').click()
        assert urlsplit(page.url).fragment == target, (locale, target)
        assert main.locator(f'section[id="{target}"]').count() == 1, (locale, target)



def check_heading_language_switches(page, base):
    for locale in ('en', 'zh'):
        other = 'zh' if locale == 'en' else 'en'
        page.goto(base + locale + '/usage.html')
        sections = page.locator('[role="main"] section[data-sw-anchor]').evaluate_all("""items => items.map(section => {
            const heading = Array.from(section.children).find(e => /^H[1-6]$/.test(e.tagName));
            return {id: section.id, stable: section.dataset.swAnchor,
                    permalink: heading?.querySelector('.headerlink')?.getAttribute('href')};
        })""")
        assert len(sections) == len(USAGE_SECTIONS), (locale, sections)
        assert {section['stable'] for section in sections} == set(USAGE_SECTIONS), (locale, sections)
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

        # Old autogenerated/localized/positional aliases still refer to the same
        # content after combining each pair of propagator/function headings.
        aliases = json.loads((PROJECT / 'tools' / 'usage_anchors.json').read_text())[locale]
        for alias, target in aliases.items():
            page.goto(base + locale + '/usage.html?from=old-heading#' + quote(alias))
            assert page.locator(f'[id="{alias}"]').evaluate('e => e.closest("section[data-sw-anchor]").dataset.swAnchor') == target
            open_mobile_menu(page)
            page.locator(f'[data-sw-language="{other}"]').click()
            assert urlsplit(page.url).path == f'/{other}/usage.html'
            assert unquote(urlsplit(page.url).fragment) == target
            assert urlsplit(page.url).query == 'from=old-heading'

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
    suites = (('api_redirects.json', 'usage.html'),
              ('modeling_redirects.json', 'modeling/wave-propagation.html'))
    for filename, target_relative in suites:
        redirects = json.loads((PROJECT / 'tools' / filename).read_text())
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
                        page.wait_for_url(lambda url: urlsplit(str(url)).path == f'/{locale}/{target_relative}', wait_until='load')
                        parsed = urlsplit(page.url)
                        # Sphinx may normalize a query space from %20 to + on load.
                        assert parse_qsl(parsed.query, keep_blank_values=True) == parse_qsl(query, keep_blank_values=True), (source, page.url)
                        assert unquote(parsed.fragment) == target, (source, page.url, target)
                        assert documents == ['/' + source_path, f'/{locale}/{target_relative}'], (source, documents)
                        assert page.locator(f'[id="{target}"]').count() == 1, (source, target)
                    finally:
                        page.remove_listener('request', record)




def check_homepage(page, base, locale, output):
    page.goto(base + locale + '/index.html', wait_until='networkidle')
    page.evaluate('document.fonts.ready')
    if locale == 'zh':
        assert page.evaluate('document.fonts.check(\'400 16px "StarWave Home Sans"\', "波")')
        assert page.evaluate('document.fonts.check(\'700 16px "StarWave Home Sans"\', "波")')
        fonts = page.evaluate('Array.from(document.fonts).filter(f => f.family.includes("StarWave Home Sans")).map(f => f.status)')
        assert fonts == ['loaded', 'loaded'], fonts
    main = page.locator('[role="main"]')
    assert main.locator('h1').count() == 1
    assert main.locator('table, .sw-api-overview, .important').count() == 0
    assert main.locator('.toctree-wrapper a').count() == 0
    logo = main.locator('.sw-home-logo')
    assert logo.count() == 1 and logo.get_attribute('alt') == 'StarWave'
    assert logo.evaluate('e => e.complete && e.naturalWidth === 950 && e.naturalHeight === 645')
    box = logo.bounding_box()
    assert box and 150 <= box['width'] <= 240, box
    assert abs(box['width'] / box['height'] - 950 / 645) < .01, box
    sidebar_logo = page.locator('.sw-sidebar-brand img')
    assert sidebar_logo.evaluate('e => e.complete && e.naturalWidth === 512')
    assert page.locator('link[rel="shortcut icon"]').get_attribute('href').endswith('starwave-favicon.png')
    capabilities = main.locator('.sw-home-capabilities > div')
    assert capabilities.count() == 3
    assert main.locator('.sw-capability-icon').count() == 3
    assert main.locator('.sw-tutorial-icon').count() == 3
    assert main.locator('.sw-home-footnote, .sw-home-number').count() == 0
    assert main.locator('.sw-home-next p').count() == 1 and main.locator('.sw-home-next a').count() == 3
    assert main.locator('.sw-home-next a[href="wsl.html"]').inner_text() == 'Windows / WSL →'
    assert all(size >= 16 for size in main.locator('.sw-home-capabilities p, .sw-home-capabilities a, .sw-path-copy strong, .sw-path-copy > span, .sw-home-next p, .sw-home-next a').evaluate_all('items => items.map(e => parseFloat(getComputedStyle(e).fontSize))'))
    assert main.locator('.sw-home-code pre').count() == 1
    assert 'loss.backward()' in main.locator('.sw-home-code').inner_text()
    assert main.locator('.sw-home-paths a').count() == 3
    actions = main.locator('.sw-home-actions a')
    assert actions.evaluate_all("items => items.map(e => e.getAttribute('href'))") == ['installation.html', 'quickstart.html', 'usage.html']
    for button in main.locator('.sw-button').all():
        assert button.bounding_box()['height'] >= 44
    typography = main.locator('.sw-home-lead, .sw-home-hero h2, .sw-home-capabilities h3').evaluate_all("items => items.map(e => getComputedStyle(e).fontFamily)")
    if locale == 'zh':
        assert len(set(typography)) == 1 and 'sans-serif' in typography[0], typography
    else:
        body_font = main.locator('.sw-home-lead').evaluate('e => getComputedStyle(e).fontFamily')
        heading_font = main.locator('.sw-home-hero h2').evaluate('e => getComputedStyle(e).fontFamily')
        feature_font = main.locator('.sw-home-capabilities h3').first.evaluate('e => getComputedStyle(e).fontFamily')
        assert 'Georgia' in body_font and feature_font == body_font, (body_font, feature_font)
        assert 'sans-serif' in heading_font and 'Georgia' not in heading_font, heading_font
        assert 'monospace' in main.locator('.sw-home-code pre').evaluate('e => getComputedStyle(e).fontFamily')
    assert main.locator('.sw-home-hero h2').inner_text() == ('波动物理，\n自动微分。' if locale == 'zh' else 'Wave Physics.\nAutomatic Differentiation.')
    assert main.locator('.sw-home-hero').evaluate("e => getComputedStyle(e).animationName === 'none'")
    if page.viewport_size['width'] <= 600 or 769 <= page.viewport_size['width'] <= 1024:
        tops = capabilities.evaluate_all('items => items.map(e => e.getBoundingClientRect().top)')
        assert tops == sorted(set(tops)), tops
    else:
        tops = capabilities.evaluate_all('items => items.map(e => e.getBoundingClientRect().top)')
        assert max(tops) - min(tops) <= 1, tops
    # Verify contrast for the custom homepage's solid text/background pairs.
    assert page.evaluate("""() => {
        function linear(v) { v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4; }
        function luminance(s) { const c = s.match(/[0-9.]+/g).slice(0,3).map(Number).map(linear); return .2126*c[0] + .7152*c[1] + .0722*c[2]; }
        return Array.from(document.querySelectorAll('.sw-home-kicker, .sw-home-lead, .sw-home-capabilities p, .sw-home-capabilities a, .sw-home-code-label, .sw-home-code-note, .sw-home-next p, .sw-home-next a, .sw-home-identity p, .sw-path-copy strong, .sw-path-copy > span, .sw-button')).every(e => {
            const style = getComputedStyle(e);
            const fg = luminance(style.color);
            const bg = e.classList.contains('sw-button-primary') ? luminance(style.backgroundColor) : 1;
            return (Math.max(fg,bg)+.05)/(Math.min(fg,bg)+.05) >= 4.5;
        });
    }"""), f'Homepage contrast failed: {locale}'
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Homepage overflow: {locale}'
    width = page.viewport_size['width']
    page.screenshot(path=str(output / f'{locale}-homepage-{width}.png'))
    page.screenshot(path=str(output / f'{locale}-homepage-full-{width}.png'), full_page=True)
    # Core buttons must navigate and retain normal Back behavior.
    for target in ('installation', 'quickstart', 'usage'):
        page.locator(f'.sw-home-actions a[href="{target}.html"]').click()
        assert urlsplit(page.url).path == f'/{locale}/{target}.html'
        page.go_back()
        assert urlsplit(page.url).path == f'/{locale}/index.html'
    other = 'zh' if locale == 'en' else 'en'
    page.goto(base + locale + '/index.html?from=home#home-workflow')
    open_mobile_menu(page)
    page.locator(f'[data-sw-language="{other}"]').click()
    assert urlsplit(page.url).path == f'/{other}/index.html'
    assert urlsplit(page.url).fragment == 'sw-section-2'
    assert urlsplit(page.url).query == 'from=home'


def check_tutorials(page, base, locale, output):
    counts = {'installation': 2, 'modeling/gradient': 4, 'inversion/fwi': 6}
    for chapter, count in counts.items():
        page.goto(base + locale + '/' + chapter + '.html', wait_until='networkidle')
        figures = page.locator('figure.sw-science-figure')
        assert figures.count() == count, (locale, chapter)
        for figure in figures.all():
            img = figure.locator('img')
            assert img.get_attribute('alt'), (locale, chapter)
            assert img.evaluate('e => e.complete && e.naturalWidth > 500'), (locale, chapter)
            assert figure.locator('figcaption').inner_text().strip(), (locale, chapter)
            assert figure.bounding_box()['width'] <= page.locator('[role="main"]').bounding_box()['width'], (locale, chapter)
        assert page.locator('a[download][href$=".ipynb"]').count() == 1, (locale, chapter)
        download_url = page.locator('a[download][href$=".ipynb"]').get_attribute('href')
        response = page.request.get(__import__('urllib.parse', fromlist=['urljoin']).urljoin(page.url, download_url))
        assert response.ok, (locale, chapter)
        notebook = response.json()
        assert all(not c.get('outputs') and c.get('execution_count') is None for c in notebook['cells'] if c['cell_type'] == 'code')
        assert '0.1.0.dev9' in page.locator('[role="main"]').inner_text()
        if chapter == 'inversion/fwi':
            text = page.locator('[role="main"]').inner_text()
            assert '91.51%' in text and '0.86%' in text and '30.639' in text
            assert page.locator('#fwi-metrics').count() == 1
            page.locator('#fwi-metrics').evaluate("e => e.scrollIntoView({block:'start'})")
            page.screenshot(path=str(output / f'{locale}-fwi-metrics-{page.viewport_size["width"]}.png'))
        if chapter == 'installation':
            page.evaluate('window.scrollTo(0, 0)')
            open_mobile_menu(page)
            page.screenshot(path=str(output / f'{locale}-search-icon-{page.viewport_size["width"]}.png'))
            close_mobile_menu(page)
    # Old localized and positional bookmarks keep their original meaning.
    old = {'id1': 'fwi-wiring', 'id2': 'fwi-loop', 'id3': 'fwi-limits'} if locale == 'zh' else {
        'run-one-update': 'fwi-wiring', 'key-points-for-the-loop': 'fwi-loop', 'validation-limits': 'fwi-limits'}
    page.goto(base + locale + '/inversion/fwi.html')
    for alias, target in old.items():
        assert page.locator(f'[id="{alias}"]').evaluate('e => e.closest("section").id') == target
    for alias, target in {'sw-section-1': 'fwi-wiring', 'sw-section-2': 'fwi-loop', 'sw-section-3': 'fwi-limits'}.items():
        assert page.locator(f'[id="{alias}"]').evaluate('e => e.closest("section").id') == target


def check_forward_modeling(page, base, locale, output):
    width = page.viewport_size['width']
    page.goto(base + locale + '/modeling/wave-propagation.html', wait_until='networkidle')
    targets = ['wave-propagation', 'wave-vrz', 'wave-vrz-details', 'wave-vrz-limits',
               'wave-vti', 'wave-vti-details', 'wave-vti-limits']
    assert page.locator('[role="main"] section[data-sw-anchor]').evaluate_all('items => items.map(e => e.dataset.swAnchor)') == targets
    toc = page.locator('.sw-page-toc')
    assert toc.locator('a').evaluate_all('items => items.map(e => e.getAttribute("href"))') == ['#' + target for target in targets[1:]]
    assert 'starwave.vrz' in page.locator('#wave-vrz').inner_text()
    assert 'starwave.vti' in page.locator('#wave-vti').inner_text()
    page.screenshot(path=str(output / f'{locale}-wave-propagation-{width}.png'))
    for target in targets[1:]:
        page.goto(base + locale + '/modeling/wave-propagation.html?from=modeling#' + target)
        other = 'en' if locale == 'zh' else 'zh'
        open_mobile_menu(page)
        page.locator(f'[data-sw-language="{other}"]').click()
        assert urlsplit(page.url).path == f'/{other}/modeling/wave-propagation.html'
        assert urlsplit(page.url).fragment == target and urlsplit(page.url).query == 'from=modeling'
    page.goto(base + locale + '/modeling/acquisition.html', wait_until='networkidle')
    figure = page.locator('figure img')
    assert figure.count() == 1 and figure.get_attribute('alt')
    assert figure.evaluate('e => e.complete && e.naturalWidth > 0')
    assert figure.bounding_box()['width'] <= page.locator('[role="main"]').bounding_box()['width']
    assert page.locator('.highlight-python').count() == 2
    figure.evaluate("e => e.scrollIntoView({block:'center'})")
    page.screenshot(path=str(output / f'{locale}-acquisition-{width}.png'))
    page.goto(base + locale + '/docker.html', wait_until='networkidle')
    main = page.locator('[role="main"]')
    assert main.locator('svg[role="img"] title').count() == 1
    assert ('尚未发布' if locale == 'zh' else 'Unreleased') in main.inner_text()
    assert main.locator('pre').count() == 0
    assert main.locator('a[href="installation.html"]').count() >= 1
    page.screenshot(path=str(output / f'{locale}-docker-{width}.png'))


def check_presentation(page, base, locale, output):
    """Check the same-origin viewer plus usable mobile and no-JS alternatives."""
    page.goto(base + locale + '/presentation.html', wait_until='networkidle')
    viewer = page.locator('object.sw-pdf-viewer')
    assert viewer.count() == 1 and viewer.get_attribute('type') == 'application/pdf'
    assert viewer.get_attribute('title') and viewer.get_attribute('aria-describedby') == 'pdf-viewer-help'
    expected_path = f'/{locale}/_static/presentations/starwave-presentation.pdf'
    source = viewer.evaluate('e => e.data')
    assert urlsplit(source).path == expected_path and urlsplit(source).netloc == urlsplit(base).netloc
    assert viewer.locator('a').count() == 2, 'Missing native object fallback links'
    opener = page.locator('[data-sw-pdf-open]')
    download = page.locator('[data-sw-pdf-download]')
    for link in (opener, download):
        assert link.is_visible() and urlsplit(link.evaluate('e => e.href')).path == expected_path
        assert link.bounding_box()['height'] >= 44
    assert opener.get_attribute('target') == '_blank' and 'noopener' in opener.get_attribute('rel').split()
    assert download.get_attribute('download') == 'StarWave-Presentation.pdf'
    response = page.request.get(base.rstrip('/') + expected_path)
    assert response.ok and 'application/pdf' in response.headers.get('content-type', '')
    manifest = json.loads((PROJECT / 'docs/_static/presentations/manifest.json').read_text())
    assert hashlib.sha256(response.body()).hexdigest() == manifest['sha256']
    assert not response.headers.get('content-disposition', '').lower().startswith('attachment')
    with page.expect_download() as event:
        download.click()
    item = event.value
    assert item.suggested_filename == 'StarWave-Presentation.pdf' and item.failure() is None
    assert hashlib.sha256(Path(item.path()).read_bytes()).hexdigest() == manifest['sha256']
    with page.expect_popup() as event:
        opener.click()
    popup = event.value
    popup.wait_for_url('**/starwave-presentation.pdf')
    assert urlsplit(popup.url).path == expected_path
    popup.close()
    width = page.viewport_size['width']
    if width <= 768:
        assert not viewer.is_visible() and page.locator('.sw-pdf-mobile').is_visible()
    else:
        assert viewer.is_visible() and not page.locator('.sw-pdf-mobile').is_visible()
        box = viewer.bounding_box()
        assert box['height'] >= 540 and box['width'] > 200
        assert box['x'] + box['width'] <= width
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.evaluate('window.scrollTo(0, 0)')
    page.screenshot(path=str(output / f'{locale}-presentation-{width}.png'), full_page=True)


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
        enter_results = {urlsplit(link).path for link in links}
        page.goto(base + locale + '/index.html')
        open_mobile_menu(page)
        page.locator('#rtd-search-form input[name="q"]').fill('illumination')
        page.locator('.sw-search-submit').click()
        page.wait_for_url(f'**/{locale}/search.html?*')
        page.wait_for_function('document.querySelectorAll("#search-results li").length > 0')
        page.wait_for_function("expected => document.querySelector('.search-summary')?.textContent.includes(expected)", arg=expected_status)
        click_results = page.locator('#search-results li a').evaluate_all('(items) => items.map(a => a.href)')
        assert enter_results == {urlsplit(link).path for link in click_results}, (locale, click_results)
        index = page.evaluate('Search._index')
        assert 'usage' in index['docnames'] and not any(name.startswith('api/') for name in index['docnames']), index['docnames']
        assert {'modeling/wave-propagation', 'modeling/acquisition', 'docker', 'presentation'} <= set(index['docnames'])
        assert not {'modeling/vrz', 'modeling/vti'} & set(index['docnames'])
        # Sphinx groups dotted functions under starwave.common rather than
        # starwave. Preserve all original search assertions and cover both.
        objects = {
            '.'.join(filter(None, (prefix.removeprefix('starwave').lstrip('.'), item[4]))): item[0]
            for prefix, items in index['objects'].items()
            if prefix == 'starwave' or prefix.startswith('starwave.')
            for item in items
        }
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
            # Full Chromium's new headless mode includes the native PDF viewer;
            # Playwright's default headless shell does not mirror that UI.
            options = {'executable_path': browser_path} if browser_path else {'channel': 'chromium'}
            browser = playwright.chromium.launch(**options)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            for size in ({'width': 1440, 'height': 1000}, {'width': 1180, 'height': 760}, {'width': 390, 'height': 844}):
                page.set_viewport_size(size)
                for locale in ('zh', 'en'):
                    for path in sorted((root / locale).rglob('*.html')):
                        if path.relative_to(root / locale).parts[0] == 'api' or path.relative_to(root / locale).as_posix() in {'modeling/vrz.html', 'modeling/vti.html'}:
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
                    check_tutorials(page, base, locale, output)
                    check_homepage(page, base, locale, output)
                    check_forward_modeling(page, base, locale, output)
                    check_presentation(page, base, locale, output)
                check_search(page, base)
            # Narrow phones and the smallest desktop sidebar layouts exercise
            # the homepage breakpoints without duplicating the entire API suite.
            for size in ({'width': 320, 'height': 740}, {'width': 820, 'height': 900}, {'width': 1024, 'height': 900}):
                page.set_viewport_size(size)
                for locale in ('zh', 'en'):
                    check_homepage(page, base, locale, output)
                    page.goto(base + locale + '/usage.html', wait_until='networkidle')
                    check_usage(page, locale, output)
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (locale, size)
                    open_mobile_menu(page)
                    check_navigation(page, locale, 'usage.html')
                    close_mobile_menu(page)
                    check_forward_modeling(page, base, locale, output)
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (locale, size)
            page.set_viewport_size({'width': 390, 'height': 844})
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
            for target in ('elastic', 'elastic-conversions', 'prepare-elastic'):
                plain.locator(f'.sw-page-toc a[href="#{target}"]').click()
                assert urlsplit(plain.url).fragment == target
            plain.locator('[data-sw-language="zh"]').click()
            assert urlsplit(plain.url).path == '/zh/usage.html'
            for prefix in ('', 'zh/', 'en/'):
                for name in ('index', 'scalar', 'vrz', 'vti'):
                    plain.goto(base + f'{prefix}api/{name}.html')
                    plain.locator('[data-sw-language="en"]').click()
                    assert urlsplit(plain.url).path == '/en/usage.html'
                    assert urlsplit(plain.url).fragment == ('usage' if name == 'index' else name)
                for name in ('vrz', 'vti'):
                    plain.goto(base + f'{prefix}modeling/{name}.html')
                    plain.locator('[data-sw-language="en"]').click()
                    assert urlsplit(plain.url).path == '/en/modeling/wave-propagation.html'
                    assert urlsplit(plain.url).fragment == 'wave-' + name
            for locale in ('zh', 'en'):
                for size in ({'width': 1440, 'height': 1000}, {'width': 390, 'height': 844}):
                    plain.set_viewport_size(size)
                    check_presentation(plain, base, locale, output)
                plain.set_viewport_size({'width': 1440, 'height': 1000})
                plain.goto(base + locale + '/index.html')
                assert plain.locator('.sw-home-logo').is_visible()
                plain.locator('.sw-home-actions a[href="quickstart.html"]').click()
                assert urlsplit(plain.url).path == f'/{locale}/quickstart.html'
                plain.goto(base + locale + '/index.html')
                plain.locator('#rtd-search-form input[name="q"]').fill('gradient')
                plain.locator('.sw-search-submit').click()
                assert urlsplit(plain.url).path == f'/{locale}/search.html'
                assert dict(parse_qsl(urlsplit(plain.url).query))['q'] == 'gradient'
                # Form navigation works without JS; Sphinx result rendering still requires JS.
            context.close()
            browser.close()
        assert not errors, errors
        print('PASS: flat desktop/mobile navigation, Usage TOC and API fields, search, direct legacy routes, language anchors, history, keyboard, PDF preview/download, and no-JS fallback.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / '_build' / 'html')
    parser.add_argument('--browser-path')
    args = parser.parse_args()
    check(args.root.resolve(), args.browser_path)
