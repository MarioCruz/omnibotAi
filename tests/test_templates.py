#!/usr/bin/env python3
"""
Sanity checks for the extracted dashboard templates + static assets.

dashboard.py used to embed ~1,460 lines of HTML inline. The markup now lives in
templates/, and the per-page CSS/JS has been lifted out into static/ (dashboard
+ kids each get their own .css and .js; omni.css holds the shared reset). These
tests confirm the files exist, are non-trivial, link each other correctly, and
carry no leftover inline <style>/<script> blocks — so a bad extraction fails
loudly instead of serving a blank or unstyled page.
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, 'templates')
STATIC = os.path.join(ROOT, 'static')


def read(name):
    with open(os.path.join(TPL, name), encoding='utf-8') as f:
        return f.read()


def read_static(name):
    with open(os.path.join(STATIC, name), encoding='utf-8') as f:
        return f.read()


class Templates(unittest.TestCase):
    def test_both_templates_exist(self):
        self.assertTrue(os.path.isfile(os.path.join(TPL, 'dashboard.html')))
        self.assertTrue(os.path.isfile(os.path.join(TPL, 'kids.html')))

    def test_dashboard_is_well_formed(self):
        html = read('dashboard.html')
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('</html>', html)
        self.assertIn('socket.io', html)          # live updates depend on this

    def test_kids_template_has_its_identity(self):
        html = read('kids.html')
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('</html>', html)
        self.assertIn('Press+Start+2P', html)      # retro font = the kids UI

    def test_no_stray_python_triple_quote_leaked_in(self):
        # If the extraction grabbed the closing '"""' it would show up here.
        for name in ('dashboard.html', 'kids.html'):
            self.assertNotIn('"""', read(name))

    # --- shared stylesheet ---

    def test_shared_stylesheet_exists_and_has_reset(self):
        self.assertTrue(os.path.isfile(os.path.join(STATIC, 'omni.css')))
        self.assertIn('box-sizing: border-box', read_static('omni.css'))

    def test_both_templates_link_shared_stylesheet(self):
        for name in ('dashboard.html', 'kids.html'):
            self.assertIn('/static/omni.css', read(name))

    def test_reset_not_duplicated_in_templates(self):
        # The '* { ... }' reset now lives only in omni.css.
        for name in ('dashboard.html', 'kids.html'):
            self.assertNotIn('* { margin: 0; padding: 0;', read(name))

    # --- per-page CSS/JS extraction ---

    def test_no_inline_style_or_script_blocks(self):
        # The whole point of the extraction: templates carry no inline <style>
        # blocks and no bare inline <script> (external libs use src=).
        for name in ('dashboard.html', 'kids.html'):
            html = read(name)
            self.assertNotIn('<style', html, f'{name} still has an inline <style> block')
            self.assertNotIn('<script>', html, f'{name} still has a bare inline <script> block')

    def test_templates_link_their_page_assets(self):
        cases = {
            'dashboard.html': ('/static/dashboard.css', '/static/dashboard.js'),
            'kids.html': ('/static/kids.css', '/static/kids.js'),
        }
        for name, (css, js) in cases.items():
            html = read(name)
            self.assertIn(css, html, f'{name} does not link {css}')
            self.assertIn(js, html, f'{name} does not link {js}')

    def test_extracted_assets_exist_and_are_substantial(self):
        for name in ('dashboard.css', 'dashboard.js', 'kids.css', 'kids.js'):
            self.assertTrue(os.path.isfile(os.path.join(STATIC, name)), f'{name} missing')
            self.assertGreater(len(read_static(name).splitlines()), 20,
                               f'{name} looks too small to be the real asset')

    def test_page_css_loads_after_shared_reset(self):
        # omni.css must come before the page stylesheet so page rules win.
        for name, page_css in (('dashboard.html', '/static/dashboard.css'),
                               ('kids.html', '/static/kids.css')):
            html = read(name)
            self.assertLess(html.index('/static/omni.css'), html.index(page_css),
                            f'{name}: page CSS should load after omni.css')


if __name__ == '__main__':
    unittest.main(verbosity=2)
