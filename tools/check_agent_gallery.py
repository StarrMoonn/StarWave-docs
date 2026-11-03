"""Regression for desktop tabs and the automatic white mobile story.
Requires a running candidate; does not start a server.
"""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright, expect
ROOT = Path(__file__).resolve().parents[1]

def run(url, browser_path=None, browser_channel='chromium'):
    assets = sorted((ROOT / 'agent/assets/gallery').glob('*.svg'))
    assert len(assets) == 6
    for path in assets:
        root = ET.parse(path).getroot()
        ids = [n.attrib['id'] for n in root.iter() if 'id' in n.attrib]
        assert len(ids) == len(set(ids))
        assert not any(n.tag.endswith('script') for n in root.iter())
        assert sum(n.tag.endswith('}svg') for n in root.iter()) == 1
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path':browser_path} if browser_path else {'channel':browser_channel}))
        for width in [1440, 390, 320]:
            page = browser.new_page(viewport={'width':width,'height':1000}, reduced_motion='reduce')
            errors=[]
            page.on('pageerror',lambda err: errors.append(str(err)))
            page.goto(url,wait_until='networkidle')
            mobile=page.locator('.mobile-gallery')
            assert page.locator('.phones>.phone-figure').count()==4
            assert mobile.locator('.gallery-panel:visible').count()==3
            assert mobile.locator('.gallery-tabs, [data-gallery-prev], [data-gallery-next]').count()==0
            assert not mobile.locator('[data-gallery-play]').is_visible()
            assert mobile.evaluate('(el)=>getComputedStyle(el).backgroundColor')=='rgb(255, 255, 255)'
            assert mobile.locator('.floating-tray').count()==3
            for img in page.locator('.gallery-art img:visible').all():
                img.scroll_into_view_if_needed()
                assert img.evaluate('(img)=>img.complete && img.naturalWidth>0')
            assert mobile.locator('.floating-tray').first.evaluate('(el)=>getComputedStyle(el).animationName')=='none'
            desktop=page.locator('[data-gallery]:not(.mobile-gallery)')
            tabs=desktop.locator('[role=tab]')
            tabs.nth(0).focus();page.keyboard.press('End')
            assert tabs.nth(2).get_attribute('aria-selected')=='true'
            page.keyboard.press('ArrowRight')
            assert tabs.nth(0).get_attribute('aria-selected')=='true'
            desktop.locator('[data-gallery-next]').click()
            assert tabs.nth(1).get_attribute('aria-selected')=='true'
            for i in range(3):
                tabs.nth(i).click()
                image=desktop.locator('.gallery-panel:visible img')
                image.scroll_into_view_if_needed()
                assert image.evaluate('(img)=>img.complete && img.naturalWidth>0')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            assert not errors,errors
            page.close()
        page=browser.new_page(java_script_enabled=False,viewport={'width':390,'height':900})
        page.goto(url)
        assert page.locator('.gallery-panel:visible').count()==6
        assert page.locator('[data-gallery-play]:visible').count()==0
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.close()
        page=browser.new_page(viewport={'width':1440,'height':1200})
        page.clock.install();page.goto(url)
        mobile=page.locator('.mobile-gallery');mobile.scroll_into_view_if_needed()
        expect(mobile).to_have_class('client-gallery mobile-gallery is-enhanced is-playing')
        page.clock.run_for(6600);expect(page.locator('#mobile-connections')).to_be_visible()
        page.clock.run_for(6600);expect(page.locator('#mobile-receipt')).to_be_visible()
        page.clock.run_for(6600);expect(page.locator('#mobile-tasks')).to_be_visible()
        play=mobile.locator('[data-gallery-play]');play.click()
        expect(play).to_have_text('播放动画')
        page.clock.run_for(15000);expect(page.locator('#mobile-tasks')).to_be_visible()
        play.click();page.clock.run_for(6600);expect(page.locator('#mobile-connections')).to_be_visible()
        page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});document.dispatchEvent(new Event('visibilitychange'))")
        page.clock.run_for(15000);expect(page.locator('#mobile-connections')).to_be_visible()
        page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>false});document.dispatchEvent(new Event('visibilitychange'))")
        page.clock.run_for(6600);expect(page.locator('#mobile-receipt')).to_be_visible()
        page.emulate_media(reduced_motion='reduce')
        page.clock.run_for(100)
        expect(mobile.locator('.gallery-panel:visible')).to_have_count(3)
        expect(play).to_be_hidden()
        page.emulate_media(reduced_motion='no-preference')
        page.clock.run_for(100)
        expect(mobile.locator('.gallery-panel:visible')).to_have_count(1)
        expect(play).to_be_visible()
        page.locator('.hero h1').scroll_into_view_if_needed()
        expect(mobile).not_to_have_class('client-gallery mobile-gallery is-enhanced is-playing')
        page.clock.run_for(15000);expect(page.locator('#mobile-receipt')).to_be_visible()
        page.goto(url+'#mobile-receipt');expect(page.locator('#mobile-receipt')).to_be_visible()
        expect(play).to_have_text('播放动画')
        browser.close()
    print('PASS: white stage, original cards, no mobile mode controls, desktop tabs, 1440/390/320, images, noJS, autoplay, pause, reduced motion, visibility, deep links')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:4173/agent/')
    parser.add_argument('--browser-path')
    parser.add_argument('--browser-channel',choices=('chromium','chrome'),default='chromium')
    args=parser.parse_args();run(args.url,args.browser_path,args.browser_channel)
