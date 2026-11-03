"""Focused static + browser regression for the public, illustrative client gallery.

Usage: python tools/check_agent_gallery.py --url http://127.0.0.1:4173/agent/
Requires a running preview of the built candidate. Never starts a server.
"""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

def run(url, browser_path=None, browser_channel='chromium'):
    assets = sorted((ROOT / 'agent/assets/gallery').glob('*.svg'))
    assert len(assets) == 6
    for path in assets:
        root = ET.parse(path).getroot()
        ids = [node.attrib['id'] for node in root.iter() if 'id' in node.attrib]
        assert len(ids) == len(set(ids)), (path, 'duplicate SVG ID')
        assert not any(node.tag.endswith('script') for node in root.iter())
        assert sum(node.tag.endswith('}svg') for node in root.iter()) == 1, 'Avoid nested SVG live-preview injection'
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({"executable_path": browser_path} if browser_path else {'channel': browser_channel}))
        for width in [1440, 390, 320]:
            page = browser.new_page(viewport={'width': width, 'height': 1000}, reduced_motion='reduce')
            errors = []
            page.on('pageerror', lambda err: errors.append(str(err)))
            page.goto(url, wait_until='networkidle')
            assert page.locator('.phones > .phone-figure').count() == 4
            assert page.locator('.gallery-panel').count() == 6
            for gallery in page.locator('[data-gallery]').all():
                tabs = gallery.locator('[role=tab]')
                panels = gallery.locator('[role=tabpanel]')
                assert tabs.count() == 3 and panels.count() == 3
                assert gallery.locator('.gallery-panel:visible').count() == 1
                tabs.nth(0).focus()
                page.keyboard.press('End')
                assert tabs.nth(2).get_attribute('aria-selected') == 'true'
                page.keyboard.press('ArrowRight')
                assert tabs.nth(0).get_attribute('aria-selected') == 'true'
                page.keyboard.press('ArrowLeft')
                assert tabs.nth(2).get_attribute('aria-selected') == 'true'
                page.keyboard.press('Home')
                page.keyboard.press('Space')
                assert tabs.nth(0).get_attribute('aria-selected') == 'true'
                gallery.locator('[data-gallery-next]').click()
                assert tabs.nth(1).get_attribute('aria-selected') == 'true'
                gallery.locator('[data-gallery-prev]').click()
                assert tabs.nth(0).get_attribute('aria-selected') == 'true'
                for i in range(3):
                    tabs.nth(i).click()
                    img = panels.nth(i).locator('img')
                    img.scroll_into_view_if_needed()
                    img.evaluate('(img) => img.decode()')
                    assert img.evaluate('(img) => img.complete && img.naturalWidth > 0')
                    assert gallery.locator('.gallery-panel:visible').count() == 1
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            assert not errors, errors
            page.close()
        page = browser.new_page(java_script_enabled=False, viewport={'width':390,'height':900})
        page.goto(url)
        assert page.locator('.gallery-panel:visible').count() == 6
        assert page.locator('.gallery-arrows:visible').count() == 0
        for image in page.locator('.gallery-art img').all():
            image.scroll_into_view_if_needed()
        assert page.locator('.phones > .phone-figure').count() == 4
        page.close()
        page = browser.new_page(viewport={'width':1440,'height':1000})
        page.goto(url + '#mobile-receipt')
        assert page.locator('#mobile-receipt').is_visible()
        assert page.locator('#tab-mobile-receipt').get_attribute('aria-selected') == 'true'
        page.close()
        browser.close()
    print('PASS: six SVGs, unique IDs, image loads, 1440/390/320 responsive, tabs, wraparound, Home/End/Space, arrows, no-JS, deep link, no console errors')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:4173/agent/')
    parser.add_argument('--browser-path')
    parser.add_argument('--browser-channel', choices=('chromium', 'chrome'), default='chromium')
    args = parser.parse_args()
    run(args.url, args.browser_path, args.browser_channel)
