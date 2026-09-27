#!/usr/bin/env python3
"""Fixtures for the restricted C-header attribute resolver."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "generate_capi_header", Path(__file__).with_name("generate-capi-header.py")
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class NormalLibrarySourceTests(unittest.TestCase):
    def test_unconditional_export_is_unchanged(self):
        source = '#[unsafe(no_mangle)]\npub extern "C" fn t4a_fixture() {}\n'
        self.assertEqual(generator.normal_library_source(source), source)

    def test_library_export_resolves_to_unconditional_export(self):
        for attribute in [
            '#[cfg_attr(not(test), unsafe(no_mangle))]',
            '#[cfg_attr(\n not(test),\n unsafe(no_mangle)\n)]',
        ]:
            with self.subTest(attribute=attribute):
                source = attribute + '\npub extern "C" fn t4a_fixture() {}\n'
                self.assertEqual(
                    generator.normal_library_source(source),
                    '#[unsafe(no_mangle)]\npub extern "C" fn t4a_fixture() {}\n',
                )

    def test_unknown_conditions_and_attributes_are_rejected(self):
        for attribute in [
            '#[cfg_attr(test, unsafe(no_mangle))]',
            '#[cfg_attr(not(test), unsafe(export_name = "other"))]',
            '#[cfg_attr(feature = "ffi", unsafe(no_mangle))]',
            '#[cfg_attr(not(test), unsafe(no_mangle), inline)]',
            '#[cfg_attr(test, derive(Clone))]',
        ]:
            with self.subTest(attribute=attribute):
                with self.assertRaisesRegex(ValueError, 'fixture.rs: unsupported conditional attribute'):
                    generator.normal_library_source(attribute, 'fixture.rs')


if __name__ == "__main__":
    unittest.main()
