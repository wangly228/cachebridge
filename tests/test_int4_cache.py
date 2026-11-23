import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from nanovllm.engine.kv_transfer import KVTransfer
from nanovllm.config import Config
from nanovllm.engine.cpu_prefix_cache import cache_namespace


class Int4ConfigTests(unittest.TestCase):
    def test_native_identity_and_unsupported_formats(self):
        hf = SimpleNamespace(dtype=torch.bfloat16, head_dim=128, max_position_embeddings=4096,
                             to_json_string=lambda **kw: "{}")
        with TemporaryDirectory() as model, patch("nanovllm.config.AutoConfig.from_pretrained", return_value=hf):
            native = Config(model, cpu_prefix_cache_blocks=1)
            explicit = Config(model, cpu_prefix_cache_blocks=1, cpu_prefix_cache_format="native")
            quant = Config(model, cpu_prefix_cache_blocks=1, cpu_prefix_cache_format="int4")
            self.assertEqual(cache_namespace(native), cache_namespace(explicit))
            self.assertNotEqual(cache_namespace(native), cache_namespace(quant))
            for opts in (dict(cpu_prefix_cache_format="int4"), dict(cpu_prefix_cache_format="int3")):
                with self.assertRaises(ValueError):
                    Config(model, **opts)
            for dtype, dim in ((torch.float16, 128), (torch.bfloat16, 80)):
                hf.dtype, hf.head_dim = dtype, dim
                with self.assertRaisesRegex(ValueError, "BF16 model"):
                    Config(model, cpu_prefix_cache_blocks=1, cpu_prefix_cache_format="int4")


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA KV pool")
class Int4TransferTests(unittest.TestCase):
    def test_known_nibbles_zero_groups_and_exact_decode(self):
        pool = torch.zeros(2, 2, 3, 3, 1, 64, dtype=torch.bfloat16, device="cuda")
        row = (list(range(-7, 8)) + [0]) * 2 + [0] * 32
        values = torch.tensor(row, dtype=pool.dtype, device="cuda")
        pool[:, :, 1] = values
        pool[1, :, 1] *= 4
        store = KVTransfer(pool, 1, cache_format="int4")
        self.addCleanup(store.close)
        store.save(1, 0)
        indices = (np.array(row, dtype=np.int16) + 8).astype(np.uint8)
        expected = indices[::2] | (indices[1::2] << 4)
        packed = store.slots[0]
        codes = 2 * 2 * 3 * 32
        np.testing.assert_array_equal(packed[:codes].reshape(-1, 32), np.tile(expected, (12, 1)))
        maxima = torch.from_numpy(packed[codes:]).view(torch.bfloat16).reshape(2, 2, 3, 2)
        self.assertEqual(maxima[..., 1].count_nonzero(), 0)
        self.assertTrue((maxima[0, ..., 0] == 7).all() and (maxima[1, ..., 0] == 28).all())
        store.restore(0, 2)
        self.assertTrue(torch.equal(pool[:, :, 1], pool[:, :, 2]))
        self.assertEqual(pool[:, :, 0].count_nonzero(), 0)
        self.assertEqual(store.block_bytes, codes + maxima.numel() * 2)

    def test_overlap_nonadjacent_blocks_match_independent_reference(self):
        torch.manual_seed(17)
        pool = torch.randn(2, 3, 7, 33, 2, 128, dtype=torch.bfloat16, device="cuda")
        store = KVTransfer(pool, 3, batch_writes=True, cache_format="int4")
        self.addCleanup(store.close)
        sources = [4, 0, 6]
        slots = [1, 2, 0]
        store.submit_save_many(list(zip(sources, slots))).result()
        for source, slot, target in zip(sources, slots, [1, 3, 5]):
            original = pool[:, :, source].cpu().float().reshape(-1, 32)
            maxima = original.abs().amax(dim=1, keepdim=True)
            q = (original / maxima * 7).round()
            expected = (q / 7 * maxima).bfloat16()
            store.restore(slot, target)
            self.assertTrue(torch.equal(pool[:, :, target].cpu().reshape(-1, 32), expected))
            before = store.slots[slot].copy()
            store.save(target, slot)
            np.testing.assert_array_equal(store.slots[slot], before)
