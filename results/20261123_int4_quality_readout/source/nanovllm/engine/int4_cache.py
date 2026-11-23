"""CPU prefix block wire format, version 1. The active cache stays in model dtype."""
import torch
import triton
import triton.language as tl


GROUP_SIZE = 32


def packed_bytes(elements):
    if elements % GROUP_SIZE:
        raise ValueError("int4 cache requires complete groups of 32 elements")
    return elements // 2 + 2 * (elements // GROUP_SIZE)


@triton.jit
def _block_codec(Block, Codes, Maxima, PLANE_STRIDE: tl.constexpr,
                 GROUPS_PER_PLANE: tl.constexpr, ENCODE: tl.constexpr,
                 TILE: tl.constexpr, GROUP: tl.constexpr):
    plane = tl.program_id(1)
    group = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = group < GROUPS_PER_PLANE
    # K/layer planes in the active pool can be more than 2^31 elements apart in total.
    start = plane.to(tl.int64) * PLANE_STRIDE
    block = Block + start + group[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    row = plane * GROUPS_PER_PLANE + group
    codes = Codes + row[:, None] * (GROUP // 2) + tl.arange(0, GROUP // 2)[None, :]
    if ENCODE:
        values = tl.load(block, mask=valid[:, None], other=0).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=1)
        divisor = tl.where(amax > 0, amax, 1.)
        q = tl.extra.cuda.libdevice.rint(tl.div_rn(values, divisor[:, None]) * 7.).to(tl.int32) + 8
        even, odd = tl.split(tl.reshape(q, (TILE * GROUP // 2, 2)))
        packed = tl.reshape(even | (odd << 4), (TILE, GROUP // 2))
        tl.store(codes, packed.to(tl.uint8), mask=valid[:, None])
        tl.store(Maxima + row, amax, mask=valid)
    else:
        packed = tl.load(codes, mask=valid[:, None], other=0).to(tl.int32)
        amax = tl.load(Maxima + row, mask=valid, other=0).to(tl.float32)
        q = tl.reshape(tl.join(packed & 15, packed >> 4), (TILE, GROUP)) - 8
        tl.store(block, (q.to(tl.float32) / 7.) * amax[:, None], mask=valid[:, None])


def transcode(block, storage, *, encode):
    """Read/write the selected strided [K/V, layer, token, head, dim] page directly."""
    planes, layers, tokens, heads, dim = block.shape
    groups = tokens * heads * dim // GROUP_SIZE
    code_bytes = block.numel() // 2
    maxima = storage[code_bytes:].view(torch.bfloat16)
    tile = 32
    _block_codec[(triton.cdiv(groups, tile), planes * layers)](
        block, storage, maxima, block.stride(1), groups, encode, tile, GROUP_SIZE)
