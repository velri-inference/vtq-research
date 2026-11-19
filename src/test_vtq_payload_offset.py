from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.vtq_format import VtqFile, VtqWriter


class PayloadOffset(unittest.TestCase):
    def test_leading_spaces_are_data(self):
        with TemporaryDirectory() as td:
            for leading in (0, 1, 64, 128):
                with self.subTest(leading=leading):
                    p = Path(td) / 'spaces.vtq'
                    writer = VtqWriter(p, [], reserve=512)
                    expected = b' ' * leading + b'\xf0'
                    writer.add_bytes('first', expected, 'u8', [len(expected)])
                    writer.add_bytes('last', b'\x05', 'u8', [1])
                    writer.finalize({'quantization': {'formats': []}})
                    reader = VtqFile(p)
                    try:
                        self.assertEqual(reader.data_off, writer.data_off)
                        self.assertEqual(bytes(reader.raw('first')), expected)
                        self.assertEqual(bytes(reader.raw('last')), b'\x05')
                    finally:
                        reader.mm.close()
                        reader.f.close()
