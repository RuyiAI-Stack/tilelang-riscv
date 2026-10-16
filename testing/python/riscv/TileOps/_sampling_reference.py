"""Independent scalar Philox4x32-10 reference shared by sampling tests."""


def _philox_uniform(seed: int, offset: int, row: int, draw: int = 0) -> float:
    mask = 0xFFFFFFFF
    counter = [draw, row, offset & mask, (offset >> 32) & mask]
    key = [seed & mask, (seed >> 32) & mask]
    for step in range(10):
        if step:
            key[0] = (key[0] + 0x9E3779B9) & mask
            key[1] = (key[1] + 0xBB67AE85) & mask
        a = counter[0] * 0xD2511F53
        b = counter[2] * 0xCD9E8D57
        counter = [((b >> 32) ^ counter[1] ^ key[0]) & mask, b & mask, ((a >> 32) ^ counter[3] ^ key[1]) & mask, a & mask]
    return (counter[0] >> 8) * 2.0**-24
