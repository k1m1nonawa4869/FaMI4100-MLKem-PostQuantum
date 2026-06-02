"""
ML-KEM-768 (FIPS 203) — Pure Python Implementation
=====================================================
Cụm 1: Core Math & Algorithms

Implements the full ML-KEM key encapsulation mechanism following NIST FIPS 203.
Reference: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import os
import hashlib
from hashlib import sha3_256, sha3_512, shake_128, shake_256

# ============================================================================
# 1. GLOBAL PARAMETERS (ML-KEM-768)
# ============================================================================
N = 256          # Polynomial degree
Q = 3329         # Modulus prime
K = 3            # Module dimension
ETA_1 = 2        # CBD parameter for secret/noise
ETA_2 = 2        # CBD parameter for encryption noise
DU = 10           # Compression parameter for u
DV = 4            # Compression parameter for v
ZETA = 17        # Primitive 256-th root of unity mod q

# ============================================================================
# 2. BIT UTILITIES
# ============================================================================

def bit_rev(i, k_bits):
    """Bit reversal of an unsigned k_bits-bit integer."""
    bin_i = bin(i & (2**k_bits - 1))[2:].zfill(k_bits)
    return int(bin_i[::-1], 2)

def bit_count(x):
    """Count the number of set bits (popcount)."""
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c

# Precompute NTT zetas (bit-reversed powers of the root of unity)
NTT_ZETAS = [pow(ZETA, bit_rev(i, 7), Q) for i in range(128)]
NTT_F = pow(128, -1, Q)  # Scaling factor for inverse NTT: 128^{-1} mod q

# ============================================================================
# 2b. POLYNOMIAL RING  R_q = Z_q[X] / (X^256 + 1)
# ============================================================================
# Đây là vành đa thức lõi của ML-KEM.
# Mỗi phần tử là một đa thức bậc ≤ 255 với hệ số trong Z_q = {0,…,3328}.
# Phép nhân trong vành được thực hiện modulo đa thức thương X^256 + 1,
# tức là mỗi lần X^256 xuất hiện, ta thay bằng -1.
# ============================================================================

class PolynomialRq:
    """
    Một phần tử của vành đa thức  R_q = Z_q[X] / (X^{256} + 1),  q = 3329.

    Biểu diễn nội tại: danh sách 256 hệ số [a_0, a_1, ..., a_255] ∈ Z_q^256
    sao cho phần tử ứng với đa thức  f(X) = a_0 + a_1*X + ... + a_255*X^255.

    Mọi phép toán đều tự động:
      - Rút gọn hệ số modulo q = 3329
      - Rút gọn đa thức modulo (X^256 + 1)
    """

    # Tham số vành (class-level, chia sẻ cho mọi phần tử)
    q = Q   # = 3329  (số nguyên tố)
    n = N   # = 256   (bậc)
    # Đa thức thương: X^256 + 1  (hệ số [1, 0, …, 0, 1] bậc 256)

    def __init__(self, coeffs=None):
        """
        Khởi tạo từ danh sách hệ số (tùy ý độ dài ≤ 256).
        Các hệ số bị rút gọn mod q, bậc bị rút gọn mod (X^256 + 1).
        """
        if coeffs is None:
            self._coeffs = [0] * self.n
        else:
            # Rút gọn modulo (X^256 + 1): nếu bậc ≥ 256, thì X^{256+k} = -X^k
            c = [int(x) % self.q for x in coeffs]
            reduced = [0] * self.n
            for i, ci in enumerate(c):
                exp = i % (2 * self.n)       # chu kỳ 512 của X^{256}+1
                if exp < self.n:
                    reduced[exp] = (reduced[exp] + ci) % self.q
                else:
                    reduced[exp - self.n] = (reduced[exp - self.n] - ci) % self.q
            self._coeffs = reduced

    # ── Truy cập hệ số ──────────────────────────────────────────────────────

    @property
    def coeffs(self):
        """Trả về bản sao danh sách 256 hệ số."""
        return list(self._coeffs)

    def __getitem__(self, i):
        return self._coeffs[i]

    # ── Phép toán vành ──────────────────────────────────────────────────────

    def __add__(self, other):
        """Cộng hai đa thức trong R_q: (f + g) mod q."""
        return PolynomialRq(
            [(self._coeffs[i] + other._coeffs[i]) % self.q
             for i in range(self.n)]
        )

    def __sub__(self, other):
        """Trừ hai đa thức trong R_q: (f - g) mod q."""
        return PolynomialRq(
            [(self._coeffs[i] - other._coeffs[i]) % self.q
             for i in range(self.n)]
        )

    def __neg__(self):
        return PolynomialRq([(-c) % self.q for c in self._coeffs])

    def __mul__(self, other):
        """
        Nhân hai đa thức trong R_q = Z_q[X]/(X^256+1).

        Thực hiện theo định nghĩa toán học (naive O(n^2)), rõ ràng nhất
        để minh họa: nhân bình thường rồi rút gọn modulo X^256 + 1.

        (Trong thuật toán chính, phép nhân được tăng tốc bằng NTT.)
        """
        # Bước 1: Nhân đa thức thông thường → tích có bậc ≤ 510
        product = [0] * (2 * self.n - 1)
        for i, ai in enumerate(self._coeffs):
            if ai == 0:
                continue
            for j, bj in enumerate(other._coeffs):
                product[i + j] = (product[i + j] + ai * bj) % self.q

        # Bước 2: Rút gọn modulo (X^256 + 1)
        # X^{256+k} ≡ -X^k  trong R_q
        reduced = [0] * self.n
        for i, ci in enumerate(product):
            if i < self.n:
                reduced[i] = (reduced[i] + ci) % self.q
            else:
                # X^i = X^{256+(i-256)} ≡ -X^{i-256}
                reduced[i - self.n] = (reduced[i - self.n] - ci) % self.q
        return PolynomialRq(reduced)

    def __eq__(self, other):
        return self._coeffs == other._coeffs

    # ── Hiển thị ────────────────────────────────────────────────────────────

    def __repr__(self):
        """In đa thức theo dạng toán học (chỉ hiện các hệ số ≠ 0)."""
        terms = []
        for i in range(self.n - 1, -1, -1):
            c = self._coeffs[i]
            if c == 0:
                continue
            if i == 0:
                terms.append(str(c))
            elif i == 1:
                terms.append(f"{c}*X" if c != 1 else "X")
            else:
                terms.append(f"{c}*X^{i}" if c != 1 else f"X^{i}")
        return " + ".join(terms) if terms else "0"

    def __str__(self):
        return self.__repr__()

    # ── Tiện ích ─────────────────────────────────────────────────────────────

    def degree(self):
        """Bậc của đa thức (chỉ số lớn nhất có hệ số ≠ 0)."""
        for i in range(self.n - 1, -1, -1):
            if self._coeffs[i] != 0:
                return i
        return -1  # đa thức zero

    def is_zero(self):
        return all(c == 0 for c in self._coeffs)

    @classmethod
    def zero(cls):
        """Phần tử zero của vành."""
        return cls([0] * cls.n)

    @classmethod
    def one(cls):
        """Phần tử đơn vị của vành."""
        c = [0] * cls.n
        c[0] = 1
        return cls(c)

    @classmethod
    def monomial(cls, degree, coeff=1):
        """Tạo đơn thức coeff * X^degree trong R_q."""
        c = [0] * (degree + 1)
        c[degree] = coeff % cls.q
        return cls(c)

    @classmethod
    def from_flat(cls, flat_list):
        """Chuyển list phẳng (dùng trong NTT) → PolynomialRq."""
        return cls(flat_list)

    def to_flat(self):
        """Chuyển PolynomialRq → list phẳng (dùng trong NTT)."""
        return list(self._coeffs)

    # ── Thông tin vành ──────────────────────────────────────────────────────

    @classmethod
    def ring_description(cls):
        return (
            f"R_q = Z_{cls.q}[X] / (X^{cls.n} + 1)\n"
            f"  Trường hữu hạn: Z_{cls.q}  (q = {cls.q} là số nguyên tố)\n"
            f"  Đa thức thương: X^{cls.n} + 1  (bất khả qui trên Z_{cls.q})\n"
            f"  Số phần tử: q^n = {cls.q}^{cls.n}  (một không gian vector khổng lồ)\n"
            f"  Căn nguyên thủy bậc {cls.n}: ζ = {ZETA}  "
            f"(vì {ZETA}^{cls.n} ≡ {pow(ZETA, cls.n, cls.q)} mod {cls.q})"
        )


# ============================================================================
# 3. HASH / XOF / PRF FUNCTIONS (Section 4 of FIPS 203)
# ============================================================================

def H(s: bytes) -> bytes:
    """SHA3-256: variable-length input → 32-byte output."""
    return sha3_256(s).digest()

def G(s: bytes) -> tuple:
    """SHA3-512: variable-length input → two 32-byte outputs."""
    h = sha3_512(s).digest()
    return h[:32], h[32:]

def J(s: bytes) -> bytes:
    """SHAKE256: variable-length input → 32-byte output."""
    return shake_256(s).digest(32)

def PRF(eta: int, s: bytes, b: int) -> bytes:
    """SHAKE256 PRF: (32-byte seed, 1-byte index) → 64*eta bytes."""
    return shake_256(s + bytes([b])).digest(64 * eta)

def XOF(rho: bytes, i: int, j: int) -> bytes:
    """SHAKE128 XOF: (32-byte seed, 2 index bytes) → 840 bytes."""
    return shake_128(rho + bytes([j]) + bytes([i])).digest(840)

# ============================================================================
# 4. ENCODE / DECODE (Algorithms 4 & 5 of FIPS 203)
# ============================================================================

def byte_encode(coeffs, d):
    """Encode a list of 256 d-bit integers into 32*d bytes (little-endian bit packing)."""
    t = 0
    for i in range(255):
        t |= coeffs[256 - i - 1]
        t <<= d
    t |= coeffs[0]
    return t.to_bytes(32 * d, "little")

def byte_decode(input_bytes, d):
    """Decode 32*d bytes into a list of 256 d-bit integers."""
    m = Q if d == 12 else (1 << d)
    coeffs = [0] * 256
    b_int = int.from_bytes(input_bytes, "little")
    mask = (1 << d) - 1
    for i in range(256):
        coeffs[i] = (b_int & mask) % m
        b_int >>= d
    return coeffs

# ============================================================================
# 5. COMPRESS / DECOMPRESS (Section 4.7 of FIPS 203)
# ============================================================================

def compress_int(x, d):
    """Compress: round((2^d / q) * x) mod 2^d."""
    t = 1 << d
    return ((t * x + 1664) // Q) % t  # 1664 = Q // 2

def decompress_int(x, d):
    """Decompress: round((q / 2^d) * x)."""
    t = 1 << (d - 1)
    return (Q * x + t) >> d

def compress_poly(coeffs, d):
    return [compress_int(c, d) for c in coeffs]

def decompress_poly(coeffs, d):
    return [decompress_int(c, d) for c in coeffs]

# ============================================================================
# 6. SAMPLING ALGORITHMS (Algorithms 6 & 7 of FIPS 203)
# ============================================================================

def sample_ntt(input_bytes):
    """
    Algorithm 6 — SampleNTT (rejection sampling).
    Parses a byte stream into 256 coefficients in [0, q).
    Output is already in NTT domain.
    """
    i, j = 0, 0
    coeffs = [0] * N
    while j < N:
        d1 = input_bytes[i] + 256 * (input_bytes[i + 1] % 16)
        d2 = (input_bytes[i + 1] // 16) + 16 * input_bytes[i + 2]
        if d1 < Q:
            coeffs[j] = d1
            j += 1
        if d2 < Q and j < N:
            coeffs[j] = d2
            j += 1
        i += 3
    return coeffs

def sample_poly_cbd(input_bytes, eta):
    """
    Algorithm 7 — SamplePolyCBD.
    Samples a polynomial from the Centered Binomial Distribution D_eta.
    """
    assert len(input_bytes) == 64 * eta
    coeffs = [0] * 256
    b_int = int.from_bytes(input_bytes, "little")
    mask = (1 << eta) - 1
    mask2 = (1 << (2 * eta)) - 1
    for i in range(256):
        x = b_int & mask2
        a = bit_count(x & mask)
        b = bit_count((x >> eta) & mask)
        b_int >>= 2 * eta
        coeffs[i] = (a - b) % Q
    return coeffs

# ============================================================================
# 7. NUMBER-THEORETIC TRANSFORM (Algorithms 9 & 10 of FIPS 203)
# ============================================================================

def ntt(coeffs):
    """
    Algorithm 9 — NTT.
    Converts polynomial from standard form to NTT representation.
    Input: 256 coefficients in Z_q. Output: 256 NTT coefficients in Z_q.
    """
    f = list(coeffs)  # copy
    k_idx = 1
    length = 128
    while length >= 2:
        start = 0
        while start < 256:
            zeta = NTT_ZETAS[k_idx]
            k_idx += 1
            for j in range(start, start + length):
                t = (zeta * f[j + length]) % Q
                f[j + length] = (f[j] - t) % Q
                f[j] = (f[j] + t) % Q
            start += 2 * length
        length >>= 1
    return f

def inv_ntt(coeffs):
    """
    Algorithm 10 — NTT⁻¹ (Inverse NTT).
    Converts NTT representation back to standard polynomial form.
    """
    f = list(coeffs)  # copy
    length = 2
    k_idx = 127
    while length <= 128:
        start = 0
        while start < 256:
            zeta = NTT_ZETAS[k_idx]
            k_idx -= 1
            for j in range(start, start + length):
                t = f[j]
                f[j] = (t + f[j + length]) % Q
                f[j + length] = (zeta * (f[j + length] - t)) % Q
            start += 2 * length
        length <<= 1
    for j in range(256):
        f[j] = (f[j] * NTT_F) % Q
    return f

# ============================================================================
# 8. NTT MULTIPLICATION (Algorithm 11 of FIPS 203)
# ============================================================================

def base_case_multiply(a0, a1, b0, b1, gamma):
    """Multiply two degree-1 polynomials modulo (X^2 - gamma)."""
    r0 = (a0 * b0 + gamma * a1 * b1) % Q
    r1 = (a1 * b0 + a0 * b1) % Q
    return r0, r1

def multiply_ntts(f_hat, g_hat):
    """
    Algorithm 11 — MultiplyNTTs.
    Coefficient-wise multiplication of two NTT-domain polynomials.
    """
    h = [0] * 256
    for i in range(64):
        z = NTT_ZETAS[64 + i]
        r0, r1 = base_case_multiply(
            f_hat[4*i], f_hat[4*i+1], g_hat[4*i], g_hat[4*i+1], z)
        r2, r3 = base_case_multiply(
            f_hat[4*i+2], f_hat[4*i+3], g_hat[4*i+2], g_hat[4*i+3], (-z) % Q)
        h[4*i], h[4*i+1], h[4*i+2], h[4*i+3] = r0, r1, r2, r3
    return h

def add_polys(a, b):
    """Element-wise addition of two polynomials mod q."""
    return [(a[i] + b[i]) % Q for i in range(256)]

def sub_polys(a, b):
    """Element-wise subtraction of two polynomials mod q."""
    return [(a[i] - b[i]) % Q for i in range(256)]

# ============================================================================
# 9. MATRIX / VECTOR OPERATIONS (over NTT-domain polynomials)
# ============================================================================

def mat_vec_mul_ntt(A_hat, s_hat):
    """Multiply k×k matrix A_hat by k-vector s_hat in NTT domain → k-vector."""
    result = []
    for i in range(K):
        acc = [0] * 256
        for j in range(K):
            prod = multiply_ntts(A_hat[i][j], s_hat[j])
            acc = add_polys(acc, prod)
        result.append(acc)
    return result

def vec_dot_ntt(t_hat, y_hat):
    """Dot product of two k-vectors in NTT domain → single polynomial."""
    acc = [0] * 256
    for i in range(K):
        prod = multiply_ntts(t_hat[i], y_hat[i])
        acc = add_polys(acc, prod)
    return acc

def transpose_matrix(A):
    """Transpose a k×k matrix."""
    return [[A[j][i] for j in range(K)] for i in range(K)]

# ============================================================================
# 10. K-PKE: BASE PUBLIC-KEY ENCRYPTION (Algorithms 13, 14, 15)
# ============================================================================

def k_pke_keygen(d: bytes):
    """
    Algorithm 13 — K-PKE.KeyGen.
    Input: 32-byte seed d.
    Output: (ek_pke, dk_pke) as byte strings.
    """
    rho, sigma = G(d + bytes([K]))

    # Generate matrix A_hat ∈ (Z_q^256)^{k×k} in NTT domain
    A_hat = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            xof_bytes = XOF(rho, i, j)
            A_hat[i][j] = sample_ntt(xof_bytes)

    # Generate secret vector s and error vector e
    n_counter = 0
    s = []
    for i in range(K):
        s.append(sample_poly_cbd(PRF(ETA_1, sigma, n_counter), ETA_1))
        n_counter += 1
    e = []
    for i in range(K):
        e.append(sample_poly_cbd(PRF(ETA_1, sigma, n_counter), ETA_1))
        n_counter += 1

    # NTT transform
    s_hat = [ntt(s[i]) for i in range(K)]
    e_hat = [ntt(e[i]) for i in range(K)]

    # t_hat = A_hat · s_hat + e_hat
    As = mat_vec_mul_ntt(A_hat, s_hat)
    t_hat = [add_polys(As[i], e_hat[i]) for i in range(K)]

    # Encode
    ek_pke = b""
    for i in range(K):
        ek_pke += byte_encode(t_hat[i], 12)
    ek_pke += rho

    dk_pke = b""
    for i in range(K):
        dk_pke += byte_encode(s_hat[i], 12)

    return ek_pke, dk_pke

def k_pke_encrypt(ek_pke: bytes, m: bytes, rand: bytes):
    """
    Algorithm 14 — K-PKE.Encrypt.
    Input: ek_pke, 32-byte message m, 32-byte randomness rand.
    Output: ciphertext bytes.
    """
    # Unpack ek
    t_hat = []
    for i in range(K):
        t_hat.append(byte_decode(ek_pke[384*i : 384*(i+1)], 12))
    rho = ek_pke[384*K:]

    # Regenerate A_hat^T
    A_hat_T = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            xof_bytes = XOF(rho, i, j)
            A_hat_T[j][i] = sample_ntt(xof_bytes)  # transposed

    # Sample y, e1, e2
    n_counter = 0
    y = []
    for i in range(K):
        y.append(sample_poly_cbd(PRF(ETA_1, rand, n_counter), ETA_1))
        n_counter += 1
    e1 = []
    for i in range(K):
        e1.append(sample_poly_cbd(PRF(ETA_2, rand, n_counter), ETA_2))
        n_counter += 1
    e2 = sample_poly_cbd(PRF(ETA_2, rand, n_counter), ETA_2)

    # NTT of y
    y_hat = [ntt(y[i]) for i in range(K)]

    # u = NTT^{-1}(A_hat^T · y_hat) + e1
    ATy = mat_vec_mul_ntt(A_hat_T, y_hat)
    u = [add_polys(inv_ntt(ATy[i]), e1[i]) for i in range(K)]

    # v = NTT^{-1}(t_hat^T · y_hat) + e2 + Decompress(Decode(m), 1)
    ty = vec_dot_ntt(t_hat, y_hat)
    mu = decompress_poly(byte_decode(m, 1), 1)
    v = add_polys(add_polys(inv_ntt(ty), e2), mu)

    # Compress and encode
    c1 = b""
    for i in range(K):
        c1 += byte_encode(compress_poly(u[i], DU), DU)
    c2 = byte_encode(compress_poly(v, DV), DV)

    return c1 + c2

def k_pke_decrypt(dk_pke: bytes, c: bytes):
    """
    Algorithm 15 — K-PKE.Decrypt.
    Input: dk_pke, ciphertext c.
    Output: 32-byte message.
    """
    n_bytes = K * DU * 32
    c1, c2 = c[:n_bytes], c[n_bytes:]

    # Decode and decompress u
    u = []
    chunk = DU * 32
    for i in range(K):
        u.append(decompress_poly(byte_decode(c1[chunk*i : chunk*(i+1)], DU), DU))

    # Decode and decompress v
    v = decompress_poly(byte_decode(c2, DV), DV)

    # Decode s_hat
    s_hat = []
    for i in range(K):
        s_hat.append(byte_decode(dk_pke[384*i : 384*(i+1)], 12))

    # u_hat = NTT(u)
    u_hat = [ntt(u[i]) for i in range(K)]

    # w = v - NTT^{-1}(s_hat^T · u_hat)
    su = vec_dot_ntt(s_hat, u_hat)
    w = sub_polys(v, inv_ntt(su))

    # Compress and encode
    m = byte_encode(compress_poly(w, 1), 1)
    return m

# ============================================================================
# 11. ML-KEM: FULL KEM WITH FO TRANSFORM (Algorithms 16–21)
# ============================================================================

def ml_kem_keygen_internal(d: bytes, z: bytes):
    """Algorithm 16 — ML-KEM.KeyGen_internal."""
    ek_pke, dk_pke = k_pke_keygen(d)
    ek = ek_pke
    dk = dk_pke + ek + H(ek) + z
    return ek, dk

def ml_kem_keygen():
    """
    Algorithm 19 — ML-KEM.KeyGen.
    Generates an encapsulation key (public) and decapsulation key (private).
    Returns: (ek, dk) byte strings.
    """
    d = os.urandom(32)
    z = os.urandom(32)
    return ml_kem_keygen_internal(d, z)

def ml_kem_encaps_internal(ek: bytes, m: bytes):
    """Algorithm 17 — ML-KEM.Encaps_internal."""
    K_shared, r = G(m + H(ek))
    c = k_pke_encrypt(ek, m, r)
    return K_shared, c

def ml_kem_encaps(ek: bytes):
    """
    Algorithm 20 — ML-KEM.Encaps.
    Input: encapsulation key ek.
    Returns: (shared_key, ciphertext).
    """
    # 1. Type Check
    if len(ek) != 384 * K + 32:
        print("Check fail: Encapsulation key type check failed (invalid length).")
        return None, None
        
    # 2. Modulus Check
    for i in range(K):
        chunk = ek[384*i : 384*(i+1)]
        if byte_encode(byte_decode(chunk, 12), 12) != chunk:
            print("Check fail: Encapsulation key modulus check failed.")
            return None, None

    m = os.urandom(32)
    return ml_kem_encaps_internal(ek, m)

def ml_kem_decaps(dk: bytes, c: bytes):
    """
    Algorithm 21 — ML-KEM.Decaps (with Fujisaki-Okamoto transform).
    Input: decapsulation key dk, ciphertext c.
    Returns: 32-byte shared secret key.
    """
    # 1. Ciphertext Type Check
    if len(c) != 32 * (K * DU + DV):
        print("Check fail: Ciphertext type check failed (invalid length).")
        return None
        
    # 2. Decapsulation Key Type Check
    if len(dk) != 768 * K + 96:
        print("Check fail: Decapsulation key type check failed (invalid length).")
        return None
        
    # 3. Hash Check
    ek_pke = dk[384 * K : 768 * K + 32]
    expected_h = dk[768 * K + 32 : 768 * K + 64]
    if H(ek_pke) != expected_h:
        print("Check fail: Hash check failed (decapsulation key hash mismatch).")
        return None

    # Parse dk
    dk_pke = dk[0 : 384 * K]
    ek_pke = dk[384 * K : 768 * K + 32]
    h = dk[768 * K + 32 : 768 * K + 64]
    z = dk[768 * K + 64 :]

    # Decrypt
    m_prime = k_pke_decrypt(dk_pke, c)

    # Re-derive
    K_prime, r_prime = G(m_prime + h)
    K_bar = J(z + c)

    # Re-encrypt and compare (implicit rejection)
    c_prime = k_pke_encrypt(ek_pke, m_prime, r_prime)

    if c == c_prime:
        return K_prime
    else:
        return K_bar  # Return garbage key if tampered

# ============================================================================
# 12. DEMO VÀNH ĐA THỨC R_q (dùng class PolynomialRq)
# ============================================================================

def demo_ring():
    """
    Minh họa cấu trúc vành đa thức R_q = Z_{3329}[X]/(X^{256}+1)
    bằng các ví dụ cụ thể.
    """
    print()
    print("╔" + "═"*58 + "╗")
    print("║" + " Demo Vành Đa Thức R_q = Z_3329[X] / (X^256 + 1) ".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    print()
    print("─" * 60)
    print("[1] Định nghĩa vành:")
    print()
    print(PolynomialRq.ring_description())

    print()
    print("─" * 60)
    print("[2] Tạo phần tử trong vành:")
    print()

    # f(X) = 3 + 2X + X^2
    f = PolynomialRq([3, 2, 1])
    print(f"  f = PolynomialRq([3, 2, 1])  →  f(X) = {f}")

    # g(X) = 1 + X
    g = PolynomialRq([1, 1])
    print(f"  g = PolynomialRq([1, 1])     →  g(X) = {g}")

    print()
    print("─" * 60)
    print("[3] Phép cộng trong R_q: (f + g) mod 3329")
    print()
    h = f + g
    print(f"  f + g = {h}")
    print(f"  (hệ số a_0: 3+1=4, a_1: 2+1=3, a_2: 1+0=1 → 4 + 3X + X^2)")

    print()
    print("─" * 60)
    print("[4] Phép nhân trong R_q: (f * g) mod (X^256+1) mod 3329")
    print()
    # f*g thông thường = 3 + 5X + 3X^2 + X^3, bậc < 256 nên không cần rút gọn
    fg = f * g
    print(f"  f * g = {fg}")
    print(f"  (f*g thông thường = 3 + 5X + 3X^2 + X^3, bậc < 256 → không đổi)")

    print()
    print("─" * 60)
    print("[5] Tính chất đặc trưng: X^256 ≡ -1 ≡ 3328 (mod 3329) trong R_q")
    print()
    # X^256 trong R_q là PolynomialRq với hệ số bậc 256 → bị rút gọn thành -1 = 3328
    X256 = PolynomialRq.monomial(256)     # X^256
    print(f"  X^256 mod (X^256+1) = {X256}")
    print(f"  (3328 = q-1 = -1 mod 3329, tức X^256 ≡ -1 trong vành)")

    X256_sq = X256 * X256  # X^512 = (X^256)^2 ≡ (-1)^2 = 1
    print(f"  (X^256)^2 = X^512 mod (X^256+1) = {X256_sq}")
    print(f"  → Xác nhận: X^512 ≡ 1 trong R_q ✓")

    print()
    print("─" * 60)
    print("[6] Rút gọn hệ số modulo q = 3329:")
    print()
    overflow = PolynomialRq([3330, 6658, 9987])  # = [1, 2, 0] mod 3329
    print(f"  PolynomialRq([3330, 6658, 9987])")
    print(f"  = PolynomialRq([3329+1, 2*3329, 3*3329]) mod 3329")
    print(f"  → {overflow}")

    print()
    print("─" * 60)
    print("[7] Phép nhân NTT vs. nhân trực tiếp (kiểm tra tương đương):")
    print()
    import random
    random.seed(42)
    fa = PolynomialRq([random.randint(0, Q-1) for _ in range(10)])
    ga = PolynomialRq([random.randint(0, Q-1) for _ in range(10)])

    # Naive multiply (trong PolynomialRq)
    naive_result = fa * ga

    # NTT-based multiply (hàm tối ưu)
    fa_hat = ntt(fa.to_flat())
    ga_hat = ntt(ga.to_flat())
    ntt_result = PolynomialRq(inv_ntt(multiply_ntts(fa_hat, ga_hat)))

    match = naive_result == ntt_result
    print(f"  f  = {str(fa)[:60]}...")
    print(f"  g  = {str(ga)[:60]}...")
    print(f"  f*g (naive O(n^2)) == f*g (NTT O(n log n)): {match}")
    print(f"  → NTT là phiên bản tối ưu của cùng phép nhân trong R_q ✓")

    print()
    print("─" * 60)
    print("[8] Vai trò trong KeyGen — ma trận A và tính t = As + e:")
    print()
    print("  Trong ML-KEM, ma trận A là một ma trận k×k phần tử R_q:")
    print(f"       A ∈ R_q^{{k×k}}  với k = {K}")
    print(f"  Khóa bí mật s ∈ R_q^{{k}}  (hệ số nhỏ, từ phân phối binomial)")
    print(f"  Nhiễu    e ∈ R_q^{{k}}  (hệ số nhỏ, từ phân phối binomial)")
    print()
    print("  Công khai: t = A·s + e  (nhân ma trận đa thức trong R_q)")
    print("  Bài toán MLWE: từ (A, t) không thể tìm lại s vì nhiễu e")
    print()


# ============================================================================
# 13. QUICK SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # ── Demo vành ──
    demo_ring()

    # ── KEM Self-test ──
    print()
    print("╔" + "═"*58 + "╗")
    print("║" + " ML-KEM-768 (FIPS 203) — KEM Self-Test ".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    ek, dk = ml_kem_keygen()
    print(f"\n[KeyGen]")
    print(f"  Encapsulation key (ek): {len(ek)} bytes")
    print(f"  Decapsulation key (dk): {len(dk)} bytes")
    assert len(ek) == 384 * K + 32
    assert len(dk) == 768 * K + 96
    print(f"  ✓ Key sizes correct (ek={384*K+32}, dk={768*K+96})")

    shared_key_bob, ciphertext = ml_kem_encaps(ek)
    print(f"\n[Encaps]")
    print(f"  Shared key (Bob):  {shared_key_bob.hex()[:48]}...")
    print(f"  Ciphertext:        {len(ciphertext)} bytes")

    shared_key_alice = ml_kem_decaps(dk, ciphertext)
    print(f"\n[Decaps]")
    print(f"  Shared key (Alice): {shared_key_alice.hex()[:48]}...")

    match = shared_key_alice == shared_key_bob
    print(f"\n[Verification]  Keys match: {match}")
    print("  ✓ ML-KEM-768 self-test PASSED!" if match else "  ✗ FAILED!")

