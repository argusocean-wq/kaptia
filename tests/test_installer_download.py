"""Regression tests for Kaptia's Windows installer download path."""
from __future__ import annotations

import io
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

from installer import windows_installer as wi


def make_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("kaptia-main/README.md", "Kaptia test archive")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, body: bytes, *, content_length: str | None = None):
        self.body = body
        self.status = 200
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self._stream = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self._stream.read(size)


class FakeInstaller:
    def __init__(self):
        self.ui_updates = []

    def update_ui(self, *args, **kwargs):
        self.ui_updates.append((args, kwargs))


class DownloadRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.root = root
        self.zip_path = root / "kaptia-source.zip"
        self.part_path = self.zip_path.with_suffix(".zip.part")
        self.installer = FakeInstaller()
        self.patchers = [
            mock.patch.object(wi, "ROOT", root),
            mock.patch.object(wi, "ZIP_PATH", self.zip_path),
            mock.patch.object(wi, "SOURCE_ZIP", "https://github.com/argusocean-wq/kaptia/archive/refs/heads/main.zip"),
            mock.patch.object(wi.time if hasattr(wi, "time") else __import__("time"), "sleep"),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def test_download_validates_zip_and_atomically_publishes_it(self):
        payload = make_zip_bytes()
        with mock.patch.object(wi.urllib.request, "urlopen", return_value=FakeResponse(
            payload, content_length=str(len(payload))
        )):
            wi.Installer._download_source(self.installer)

        self.assertTrue(self.zip_path.is_file())
        with zipfile.ZipFile(self.zip_path) as archive:
            self.assertIsNone(archive.testzip())
        self.assertFalse(self.part_path.exists())

    def test_http_403_falls_back_to_codeload_endpoint(self):
        payload = make_zip_bytes()
        denied = urllib.error.HTTPError(
            "https://github.com/argusocean-wq/kaptia/archive/refs/heads/main.zip",
            403, "Forbidden", {}, None
        )
        with mock.patch.object(wi.urllib.request, "urlopen",
                               side_effect=[denied, FakeResponse(payload)] ) as opener:
            wi.Installer._download_source(self.installer)

        self.assertEqual(opener.call_count, 2)
        self.assertIn("codeload.github.com", opener.call_args_list[1].args[0].full_url)
        self.assertTrue(self.zip_path.is_file())

    def test_content_length_mismatch_is_retried_and_not_published(self):
        payload = make_zip_bytes()
        with mock.patch.object(wi.urllib.request, "urlopen", side_effect=[
            FakeResponse(payload, content_length=str(len(payload) + 7)),
            FakeResponse(payload, content_length=str(len(payload))),
        ]):
            wi.Installer._download_source(self.installer)

        self.assertTrue(self.zip_path.is_file())
        with zipfile.ZipFile(self.zip_path) as archive:
            self.assertIsNone(archive.testzip())

    def test_corrupt_zip_is_not_published(self):
        with mock.patch.object(wi.urllib.request, "urlopen",
                               return_value=FakeResponse(b"not a zip file")):
            with self.assertRaises(RuntimeError):
                wi.Installer._download_source(self.installer)

        self.assertFalse(self.zip_path.exists())
        self.assertFalse(self.part_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
