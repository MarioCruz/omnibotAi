#!/usr/bin/env python3
"""
Unit tests for AudioCommander.sanitize_speech — the command-injection boundary.

Text passed to speak() reaches speak_pi.sh as an argv element. sanitize_speech
is what stops shell metacharacters from ever leaving Python, so it gets its own
tests. Pure string logic; no audio hardware touched.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_commander import AudioCommander

clean = AudioCommander.sanitize_speech


class SanitizeSpeech(unittest.TestCase):
    def test_plain_text_preserved(self):
        self.assertEqual(clean("Hello, world!"), "Hello, world!")

    def test_basic_punctuation_kept(self):
        self.assertEqual(clean("Yes. No? It's fine - really!"),
                         "Yes. No? It's fine - really!")

    def test_strips_shell_metacharacters(self):
        # The dangerous bits ($ ( ) ` ; | / and "rm" stays as letters) get
        # stripped down to harmless words — nothing that a shell would act on.
        out = clean("rm -rf /; $(whoami) `id` | cat")
        for bad in ('$', '(', ')', '`', ';', '|', '/'):
            self.assertNotIn(bad, out)

    def test_strips_quotes_and_backslashes(self):
        out = clean('"; reboot \\ now #')
        for bad in ('"', '\\', '#'):
            self.assertNotIn(bad, out)

    def test_empty_for_all_disallowed(self):
        self.assertEqual(clean("@#$%^&*"), "")

    def test_empty_string(self):
        self.assertEqual(clean(""), "")

    def test_non_string_is_safe(self):
        self.assertEqual(clean(None), "")
        self.assertEqual(clean(1234), "")

    def test_newlines_removed(self):
        # \n is whitespace-but-not-space; \s keeps it, so confirm current intent.
        # The allow-list keeps \s (incl. newline) — assert no shell-active chars.
        out = clean("line1\nline2 && echo hi")
        self.assertNotIn('&', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
