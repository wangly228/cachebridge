import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.kv_transfer import KVTransfer


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA copies and events")
class TransferTests(unittest.TestCase):
    def test_partial_host_write_drains_stream_and_stages_can_be_reused(self):
        shape = (2, 2, 6, 256, 1, 16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            values = torch.arange(2 * 2 * 6 * 256 * 16, device="cuda").remainder(251)
            cache = values.to(torch.bfloat16).reshape(shape)
            transfer = KVTransfer(cache, 4, batch_writes=True)
            self.addCleanup(transfer.close)
            pairs = [(0, 3), (2, 1), (4, 0)]

            def fail_after_first_bytes(dst, src):
                dst[:64] = src[:64]
                raise MemoryError("host copy failed with the next D2H already submitted")

            with patch("nanovllm.engine.kv_transfer.np.copyto", fail_after_first_bytes):
                with self.assertRaises(MemoryError):
                    transfer.save_many(pairs)
            self.assertTrue(stream.query())

            transfer.save_many(pairs)
            self.assertTrue(stream.query())
            for src, slot in pairs:
                transfer.restore(slot, src + 1)
                self.assertTrue(torch.equal(cache[:, :, src].view(torch.int16), cache[:, :, src + 1].view(torch.int16)))
