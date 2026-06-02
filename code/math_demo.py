"""
Minh họa Toán học Lõi của ML-KEM (FIPS 203)
==============================================
File chạy độc lập — dùng cho slide thuyết trình.

Trình bày từng bước:
  1. Trường hữu hạn Z_q (q = 3329)
  2. Vành đa thức R_q = Z_q[X]/(X^256 + 1)
  3. Phép nhân đa thức trong vành (naive + NTT)
  4. Nén/Giải nén (Compress/Decompress)
  5. Phân phối Binomial tâm hóa (CBD) — sinh nhiễu
  6. Bài toán MLWE: t = A·s + e
  7. Chạy KeyGen / Encaps / Decaps minh họa
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from ml_kem_core import (
    Q, N, K, ZETA, ETA_1, DU, DV,
    PolynomialRq,
    ntt, inv_ntt, multiply_ntts, add_polys,
    compress_int, decompress_int,
    sample_poly_cbd, PRF,
    sample_ntt, XOF,
    mat_vec_mul_ntt,
    k_pke_keygen, ml_kem_keygen, ml_kem_encaps, ml_kem_decaps,
    G, NTT_ZETAS
)

import random
random.seed(2024)

def title(text):
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + text.center(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()

def section(n, text):
    print()
    print("─" * 64)
    print(f"  [{n}] {text}")
    print("─" * 64)

# ================================================================
# 1. TRƯỜNG HỮU HẠN  Z_q  (q = 3329)
# ================================================================

def demo_finite_field():
    section(1, "Trường hữu hạn Z_q  (q = 3329)")

    print(f"""
  ML-KEM hoạt động trên trường hữu hạn Z_q với q = {Q} (số nguyên tố).
  Mọi phép tính hệ số đều thực hiện modulo {Q}.

  Ví dụ:
    1000 + 2500 = {(1000 + 2500) % Q}    (mod {Q})
    100  × 200  = {(100 * 200) % Q}   (mod {Q})
    3329 mod q  = {3329 % Q}       (q ≡ 0)
    -1   mod q  = {(-1) % Q}    (tức 3328)

  Số nguyên tố q = 3329 được chọn vì:
    • 3329 = 13 × 256 + 1  →  hỗ trợ NTT bậc 256
    • ζ = {ZETA} là căn nguyên thủy bậc 256:
      ζ^256 mod q = {pow(ZETA, 256, Q)}  (≡ 1)
      ζ^128 mod q = {pow(ZETA, 128, Q)}  (≡ -1 = q-1)
""")

# ================================================================
# 2. VÀNH ĐA THỨC  R_q = Z_q[X] / (X^256 + 1)
# ================================================================

def demo_polynomial_ring():
    section(2, "Vành đa thức R_q = Z_q[X] / (X^256 + 1)")

    print()
    print(PolynomialRq.ring_description())

    print(f"""
  Vành thương R_q = Z_q[X] / (X^256 + 1) nghĩa là:
    • Mỗi phần tử là một đa thức bậc ≤ 255 với hệ số trong Z_{Q}.
    • Phép cộng: cộng từng hệ số mod q.
    • Phép nhân: nhân đa thức thường → rút gọn mod (X^256 + 1).
      Quy tắc rút gọn: X^256 ≡ -1, nên X^{{256+k}} ≡ -X^k.

  Ví dụ tạo phần tử:""")

    f = PolynomialRq([3, 2, 1])
    g = PolynomialRq([1, 1])
    print(f"    f = PolynomialRq([3, 2, 1])  →  {f}")
    print(f"    g = PolynomialRq([1, 1])     →  {g}")

    print(f"\n  Phép cộng  f + g:")
    h = f + g
    print(f"    = {h}")

    print(f"\n  Phép trừ   f - g:")
    d = f - g
    print(f"    = {d}")

    print(f"\n  Phép nhân  f × g (trong R_q):")
    fg = f * g
    print(f"    = {fg}")
    print(f"    (Bậc < 256 nên không cần rút gọn modulo X^256+1)")

    print(f"\n  ─── Tính chất đặc trưng: X^256 ≡ -1 ─── ")
    X256 = PolynomialRq.monomial(256)
    print(f"    X^256 mod (X^256+1) = {X256}")
    print(f"    (3328 = -1 mod 3329)")

    X257 = PolynomialRq.monomial(257)
    print(f"    X^257 mod (X^256+1) = {X257}")
    print(f"    (X^257 = X·X^256 ≡ X·(-1) = -X ≡ 3328·X)")

    X512 = X256 * X256
    print(f"    X^512 mod (X^256+1) = {X512}")
    print(f"    ((-1)^2 = 1  ✓)")

    print(f"\n  ─── Rút gọn hệ số modulo q ───")
    overflow = PolynomialRq([3330, 6658, 0])
    print(f"    PolynomialRq([3330, 6658, 0])")
    print(f"    = [(3329+1) mod 3329, (2×3329) mod 3329, 0]")
    print(f"    → {overflow}")

# ================================================================
# 3. PHÉP NHÂN ĐA THỨC: NAIVE vs NTT
# ================================================================

def demo_ntt_multiplication():
    section(3, "Phép nhân đa thức: Naive O(n^2) vs NTT O(n log n)")

    print(f"""
  Nhân 2 đa thức bậc 255 theo cách thường: O(n^2) = O(65536) phép nhân.
  NTT (Number-Theoretic Transform) giảm xuống O(n log n) = O(2048).

  NTT hoạt động theo 3 bước:
    1. Chuyển f, g → miền NTT:  f̂ = NTT(f),  ĝ = NTT(g)
    2. Nhân từng hệ số:          ĥ = f̂ ⊙ ĝ    (pointwise)
    3. Chuyển ngược:             h = NTT⁻¹(ĥ)
""")

    # Tạo 2 đa thức ngẫu nhiên (bậc nhỏ để hiển thị được)
    f = PolynomialRq([random.randint(0, Q - 1) for _ in range(8)])
    g = PolynomialRq([random.randint(0, Q - 1) for _ in range(8)])
    print(f"  f = {f}")
    print(f"  g = {g}")

    # Naive
    naive = f * g
    print(f"\n  Naive multiply (f * g trong R_q):")
    print(f"    = {naive}")

    # NTT-based
    f_hat = ntt(f.to_flat())
    g_hat = ntt(g.to_flat())
    h_hat = multiply_ntts(f_hat, g_hat)
    ntt_result = PolynomialRq(inv_ntt(h_hat))

    print(f"\n  NTT multiply:")
    print(f"    f̂ = NTT(f)  →  [{f_hat[0]}, {f_hat[1]}, {f_hat[2]}, ...]  (256 hệ số NTT)")
    print(f"    ĝ = NTT(g)  →  [{g_hat[0]}, {g_hat[1]}, {g_hat[2]}, ...]")
    print(f"    ĥ = f̂ ⊙ ĝ   (nhân pointwise + BaseCaseMultiply)")
    print(f"    h = NTT⁻¹(ĥ) = {ntt_result}")

    match = naive == ntt_result
    print(f"\n  Kết quả giống nhau? {match}  ✓" if match else f"\n  ✗ SAI!")

    print(f"""
  ─── Bảng NTT Zetas (10 phần tử đầu) ───
  NTT sử dụng bảng ζ^(bit_rev(i)) mod q đã tính sẵn:
    zetas = {NTT_ZETAS[:10]}...
  Tổng cộng 128 giá trị zeta, tính 1 lần, dùng cho mọi phép nhân.
""")

# ================================================================
# 4. NÉN / GIẢI NÉN (Compress / Decompress)
# ================================================================

def demo_compress():
    section(4, "Nén / Giải nén bit (Compress / Decompress)")

    print(f"""
  Để giảm kích thước ciphertext, ML-KEM nén hệ số từ 12 bit → d bit.

  Compress_d(x) = round((2^d / q) × x) mod 2^d
  Decompress_d(y) = round((q / 2^d) × y)

  Đây là nén MẤT MÁT (lossy): Decompress(Compress(x)) ≈ x, không chính xác = x.
  Nhưng sai số đủ nhỏ để thuật toán vẫn khôi phục đúng thông điệp.

  Ví dụ với d = {DV} (dùng cho thành phần v trong ciphertext):
""")
    print(f"  {'x (gốc)':>10} {'Compress':>12} {'Decompress':>14} {'Sai số':>10}")
    print(f"  {'─'*10} {'─'*12} {'─'*14} {'─'*10}")
    for x in [0, 100, 500, 1000, 1664, 2000, 3000, 3328]:
        c = compress_int(x, 4)
        d = decompress_int(c, 4)
        err = min(abs(x - d), Q - abs(x - d))  # error mod q
        print(f"  {x:>10} {c:>12} {d:>14} {err:>10}")

    print(f"\n  Sai số tối đa ≤ q/(2^(d+1)) = {Q} / {2**(4+1)} ≈ {Q / 2**(4+1):.0f}")

# ================================================================
# 5. PHÂN PHỐI BINOMIAL TÂM HÓA (CBD)
# ================================================================

def demo_cbd():
    section(5, "Phân phối Binomial tâm hóa (CBD) — sinh s, e")

    print(f"""
  Vector bí mật s và nhiễu e được lấy mẫu từ phân phối CBD_η.
  Với ML-KEM-768: η₁ = η₂ = {ETA_1}.

  CBD_2 sinh hệ số trong {{-2, -1, 0, 1, 2}} với xác suất:
    P(-2) = 1/16,  P(-1) = 4/16,  P(0) = 6/16,  P(1) = 4/16,  P(2) = 1/16

  Hệ số nhỏ → vector s "ngắn" → bài toán MLWE khó giải.
""")

    # Sinh một đa thức từ CBD và thống kê phân phối
    seed = b'\x42' * 32
    poly = sample_poly_cbd(PRF(ETA_1, seed, 0), ETA_1)

    # Đếm phân phối (chuyển về khoảng [-2, 2])
    counts = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
    for c in poly:
        val = c if c <= Q // 2 else c - Q  # chuyển từ [0, q) về [-q/2, q/2]
        if val in counts:
            counts[val] += 1

    print(f"  Ví dụ: s = SamplePolyCBD(PRF(σ, 0), η=2)")
    print(f"  Thống kê 256 hệ số:")
    for val in sorted(counts.keys()):
        bar = "█" * (counts[val] // 3)
        print(f"    {val:+d}: {counts[val]:>3} lần  {bar}")

    print(f"\n  10 hệ số đầu tiên (biểu diễn dấu):")
    display = []
    for c in poly[:10]:
        v = c if c <= Q // 2 else c - Q
        display.append(str(v))
    print(f"    [{', '.join(display)}]")

# ================================================================
# 6. BÀI TOÁN MLWE: t = A·s + e
# ================================================================

def demo_mlwe():
    section(6, "Bài toán MLWE: t = A·s + e")

    print(f"""
  Nền tảng an ninh của ML-KEM: Module Learning With Errors (MLWE).

  Cho:
    A ∈ R_q^{{k×k}}  — ma trận ngẫu nhiên công khai (k={K})
    s ∈ R_q^{{k}}     — vector bí mật (hệ số nhỏ, từ CBD)
    e ∈ R_q^{{k}}     — vector nhiễu (hệ số nhỏ, từ CBD)
    t = A·s + e      — vector công khai

  Bài toán: Cho (A, t), tìm s.
  → Khó giải (reducible về bài toán shortest vector trên lattice).
  → Ngay cả máy tính lượng tử cũng không có thuật toán hiệu quả.
""")

    # Thực sự sinh A, s, e, t theo FIPS 203
    print("  Đang sinh khóa thực tế theo FIPS 203...")
    import os
    d = os.urandom(32)
    rho, sigma = G(d + bytes([K]))

    # Sinh A
    A_hat = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            A_hat[i][j] = sample_ntt(XOF(rho, i, j))

    # Sinh s, e
    n_counter = 0
    s_hat = []
    for i in range(K):
        poly = sample_poly_cbd(PRF(ETA_1, sigma, n_counter), ETA_1)
        s_hat.append(ntt(poly))
        n_counter += 1

    e_hat = []
    for i in range(K):
        poly = sample_poly_cbd(PRF(ETA_1, sigma, n_counter), ETA_1)
        e_hat.append(ntt(poly))
        n_counter += 1

    # t = A·s + e
    As = mat_vec_mul_ntt(A_hat, s_hat)
    t_hat = [add_polys(As[i], e_hat[i]) for i in range(K)]

    print(f"\n  Ma trận A (NTT domain) — kích thước {K}×{K}:")
    for i in range(K):
        for j in range(K):
            print(f"    A[{i}][{j}] = [{A_hat[i][j][0]}, {A_hat[i][j][1]}, {A_hat[i][j][2]}, ...] (256 hệ số)")

    print(f"\n  Vector bí mật s (NTT domain) — {K} phần tử:")
    for i in range(K):
        print(f"    s[{i}] = [{s_hat[i][0]}, {s_hat[i][1]}, {s_hat[i][2]}, ...]")

    print(f"\n  Vector công khai t = A·s + e (NTT domain) — {K} phần tử:")
    for i in range(K):
        print(f"    t[{i}] = [{t_hat[i][0]}, {t_hat[i][1]}, {t_hat[i][2]}, ...]")

    print(f"""
  → Kẻ tấn công thấy A và t, nhưng KHÔNG biết s và e.
  → Nhiễu e khiến t trông ngẫu nhiên, che giấu s hoàn toàn.
  → Đây là nền tảng "post-quantum" của ML-KEM.
""")

# ================================================================
# 7. CHẠY KEYGEN / ENCAPS / DECAPS MINH HỌA
# ================================================================

def demo_kem():
    section(7, "ML-KEM-768: KeyGen → Encaps → Decaps")

    print(f"""
  Giao thức KEM gồm 3 thuật toán:
    1. KeyGen()           → (ek, dk)         [Alice]
    2. Encaps(ek)          → (K, ciphertext)  [Bob]
    3. Decaps(dk, cipher)  → K'               [Alice]
  Nếu không bị tấn công: K = K' (cùng shared secret 256-bit).
""")

    import time

    t0 = time.perf_counter()
    ek, dk = ml_kem_keygen()
    t_kg = (time.perf_counter() - t0) * 1000
    print(f"  [KeyGen] {t_kg:.1f} ms")
    print(f"    ek (public):  {len(ek)} bytes  = {ek.hex()[:40]}...")
    print(f"    dk (private): {len(dk)} bytes  = {dk.hex()[:40]}...")

    t0 = time.perf_counter()
    shared_bob, ct = ml_kem_encaps(ek)
    t_enc = (time.perf_counter() - t0) * 1000
    print(f"\n  [Encaps] {t_enc:.1f} ms")
    print(f"    K (Bob):      {shared_bob.hex()}")
    print(f"    ciphertext:   {len(ct)} bytes  = {ct.hex()[:40]}...")

    t0 = time.perf_counter()
    shared_alice = ml_kem_decaps(dk, ct)
    t_dec = (time.perf_counter() - t0) * 1000
    print(f"\n  [Decaps] {t_dec:.1f} ms")
    print(f"    K' (Alice):   {shared_alice.hex()}")

    match = shared_bob == shared_alice
    print(f"\n  K == K' ?  {'YES — Shared secret established!' if match else 'NO — FAILED!'}")

    print(f"""
  ─── Tóm tắt kích thước ───
    Encapsulation key:  {len(ek):>5} bytes  (= 384×{K} + 32)
    Decapsulation key:  {len(dk):>5} bytes  (= 768×{K} + 96)
    Ciphertext:         {len(ct):>5} bytes  (= 32×({K}×d_u + d_v))
    Shared secret:      {len(shared_bob):>5} bytes  (256 bits)
""")

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    title("Minh Họa Toán Học Lõi — ML-KEM-768 (FIPS 203)")

    demo_finite_field()
    demo_polynomial_ring()
    demo_ntt_multiplication()
    demo_compress()
    demo_cbd()
    demo_mlwe()
    demo_kem()

    print("=" * 64)
    print("  Hoàn tất minh họa toán học.")
    print("=" * 64)
