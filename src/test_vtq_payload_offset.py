import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from src.vtq_format import MAGIC, PREFIX_BYTES, VERSION, VtqFile, VtqWriter, source_metadata


def _write_v1_file(path: Path, tensors: list[tuple[str, bytes]], reserve: int = 512) -> int:
    data_off = ((16 + reserve) + 63) // 64 * 64
    reserve = data_off - 16
    pos = 0
    payload = bytearray()
    index = {}
    for name, raw in tensors:
        pad = (-pos) & 63
        payload += b"\0" * pad
        pos += pad
        index[name] = {"dtype": "u8", "shape": [len(raw)], "offset": pos}
        payload += raw
        pos += len(raw)
    hdr = json.dumps({"quantization": {"formats": []}, "tensors": index}, separators=(",", ":")).encode("utf-8")
    assert len(hdr) <= reserve
    path.write_bytes(MAGIC + struct.pack("<IQ", 1, len(hdr)) + hdr + b" " * (reserve - len(hdr)) + bytes(payload))
    return data_off


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
                    with VtqFile(p) as reader:
                        self.assertEqual(reader.version, VERSION)
                        self.assertEqual(reader.data_off, PREFIX_BYTES)
                        self.assertEqual(reader.data_off, writer.data_off)
                        self.assertEqual(bytes(reader.raw('first')), expected)
                        self.assertEqual(bytes(reader.raw('last')), b'\x05')

    def test_v1_backward_compat_leading_spaces(self):
        with TemporaryDirectory() as td:
            for leading in (0, 1, 64, 128):
                with self.subTest(leading=leading):
                    p = Path(td) / 'v1_spaces.vtq'
                    expected = b' ' * leading + b'\xf0'
                    data_off = _write_v1_file(p, [('first', expected), ('last', b'\x05')], reserve=512)
                    with VtqFile(p) as reader:
                        self.assertEqual(reader.version, 1)
                        self.assertEqual(reader.data_off, data_off)
                        self.assertEqual(bytes(reader.raw('first')), expected)
                        self.assertEqual(bytes(reader.raw('last')), b'\x05')

    def test_unlimited_footer_metadata(self):
        with TemporaryDirectory() as td:
            p = Path(td) / 'large_meta.vtq'
            big_blob = "привет_世界_" * 600_000  # > 10 MiB UTF-8 JSON footer
            writer = VtqWriter(p, [])
            writer.add_bytes('w', b'\x01\x02\x03\x04', 'u8', [4])
            writer.finalize({
                'config': {'hidden_size': 4096},
                'chat_template': big_blob,
                'quantization': {'formats': []},
            })
            with VtqFile(p) as reader:
                self.assertEqual(reader.version, VERSION)
                self.assertEqual(reader.data_off, PREFIX_BYTES)
                self.assertGreater(p.stat().st_size - reader.header_offset, 8 << 20)
                self.assertEqual(reader.header['config'], {'hidden_size': 4096})
                self.assertEqual(reader.header['chat_template'], big_blob)
                self.assertEqual(bytes(reader.raw('w')), b'\x01\x02\x03\x04')

    def test_source_metadata_roundtrip(self):
        with TemporaryDirectory() as td:
            mdir = Path(td) / 'model'
            mdir.mkdir()
            (mdir / 'config.json').write_text('{"model_type":"qwen3","hidden_size":2048}', encoding='utf-8')
            (mdir / 'generation_config.json').write_text('{"temperature":0.6}', encoding='utf-8')
            (mdir / 'tokenizer.json').write_text('{"version":"1.0","vocab":{"привет":1}}', encoding='utf-8')
            (mdir / 'tokenizer_config.json').write_text('{"eos_token":"</s>"}', encoding='utf-8')
            (mdir / 'chat_template.jinja').write_text('{% for m in messages %}{{ m.content }}{% endfor %}', encoding='utf-8')
            meta = source_metadata(mdir)
            p = Path(td) / 'with_meta.vtq'
            writer = VtqWriter(p, [])
            writer.add_bytes('x', b'\xaa', 'u8', [1])
            writer.finalize({**meta, 'quantization': {'formats': []}})
            with VtqFile(p) as reader:
                self.assertEqual(reader.header['config']['model_type'], 'qwen3')
                self.assertEqual(reader.header['generation_config']['temperature'], 0.6)
                self.assertEqual(reader.header['tokenizer']['vocab']['привет'], 1)
                self.assertEqual(reader.header['tokenizer_config']['eos_token'], '</s>')
                self.assertEqual(reader.header['chat_template'], '{% for m in messages %}{{ m.content }}{% endfor %}')

    def test_v2_rejects_corrupt_and_legacy_v2_header_at_start(self):
        with TemporaryDirectory() as td:
            # Legacy vtq_v2_format.py wrote VERSION=2 with JSON starting at offset 16 (non-zero reserved bytes).
            legacy = Path(td) / 'legacy_v2.vtq'
            hdr = b'{"quantization":{"formats":[]},"tensors":{}}' + b' ' * 100
            legacy.write_bytes(MAGIC + struct.pack('<IQ', 2, len(hdr)) + hdr)
            with self.assertRaises(ValueError):
                VtqFile(legacy)

            # Truncated footer in v2 file
            p = Path(td) / 'trunc.vtq'
            writer = VtqWriter(p, [])
            writer.add_bytes('w', b'\x01\x02', 'u8', [2])
            writer.finalize({'quantization': {'formats': []}})
            raw = p.read_bytes()
            p.write_bytes(raw[:-4])
            with self.assertRaises(ValueError):
                VtqFile(p)
