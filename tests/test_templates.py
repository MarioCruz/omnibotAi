#!/usr/bin/env python3
"""
Sanity checks for the extracted dashboard templates.

dashboard.py used to embed ~1,460 lines of HTML inline; the markup now lives in
templates/. These tests confirm the files exist, are non-trivial, and still
carry the markers dashboard.py / the browser rely on — so a bad extraction or a
deleted file fails loudly instead of serving a blank page.
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, 'templates')


def read(name):
    with open(os.path.join(TPL, name), encoding='utf-8') as f:
        return f.read()


class Templates(unittest.TestCase):
    def test_both_templates_exist(self):
        self.assertTrue(os.path.isfile(os.path.join(TPL, 'dashboard.html')))
        self.assertTrue(os.path.isfile(os.path.join(TPL, 'kids.html')))

    def test_dashboard_is_well_formed_and_substantial(self):
        html = read('dashboard.html')
        self.assertGreater(len(html.splitlines()), 200)
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
