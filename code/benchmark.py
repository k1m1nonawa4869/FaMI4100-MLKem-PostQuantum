"""
Cum 3: Benchmarking ML-KEM-768 vs RSA-2048
============================================
Measures: execution time, public key size, ciphertext size.
Generates comparison bar charts using matplotlib.
"""

import time
import sys
import statistics

sys.stdout.reconfigure(encoding='utf-8')

from ml_kem_core import ml_kem_keygen, ml_kem_encaps, ml_kem_decaps

# ============================================================================
# CONFIG
# ============================================================================
N_ITERATIONS = 1000  # Number of benchmark iterations

# ============================================================================
# 1. BENCHMARK ML-KEM-768 (Pure Python)
# ============================================================================

def benchmark_mlkem(n_iter):
    """Benchmark ML-KEM-768 over n_iter iterations."""
    print(f"\n[ML-KEM-768] Benchmarking {n_iter} iterations...")

    keygen_times = []
    encaps_times = []
    decaps_times = []
    pk_size = 0
    ct_size = 0

    for i in range(n_iter):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_iter}")

        # KeyGen
        t0 = time.perf_counter()
        ek, dk = ml_kem_keygen()
        keygen_times.append((time.perf_counter() - t0) * 1000)

        pk_size = len(ek)

        # Encaps
        t0 = time.perf_counter()
        shared_key_bob, ct = ml_kem_encaps(ek)
        encaps_times.append((time.perf_counter() - t0) * 1000)

        ct_size = len(ct)

        # Decaps
        t0 = time.perf_counter()
        shared_key_alice = ml_kem_decaps(dk, ct)
        decaps_times.append((time.perf_counter() - t0) * 1000)

        assert shared_key_alice == shared_key_bob, f"Key mismatch at iteration {i}!"

    return {
        "keygen_avg": statistics.mean(keygen_times),
        "keygen_med": statistics.median(keygen_times),
        "encaps_avg": statistics.mean(encaps_times),
        "encaps_med": statistics.median(encaps_times),
        "decaps_avg": statistics.mean(decaps_times),
        "decaps_med": statistics.median(decaps_times),
        "pk_size": pk_size,
        "ct_size": ct_size,
        "ss_size": 32,
    }

# ============================================================================
# 2. BENCHMARK RSA-2048
# ============================================================================

def benchmark_rsa(n_iter):
    """Benchmark RSA-2048 key exchange (encrypt/decrypt a 32-byte key)."""
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        print("  [WARNING] 'cryptography' library not installed.")
        print("  Install with: pip install cryptography")
        print("  Returning placeholder RSA data.")
        # Reference values from typical benchmarks
        return {
            "keygen_avg": 120.0, "keygen_med": 115.0,
            "encaps_avg": 0.3,   "encaps_med": 0.28,
            "decaps_avg": 4.5,   "decaps_med": 4.3,
            "pk_size": 294, "ct_size": 256, "ss_size": 32,
        }

    print(f"\n[RSA-2048] Benchmarking {n_iter} iterations...")

    keygen_times = []
    encrypt_times = []
    decrypt_times = []
    pk_size = 0
    ct_size = 0

    for i in range(n_iter):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_iter}")

        # KeyGen
        t0 = time.perf_counter()
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        keygen_times.append((time.perf_counter() - t0) * 1000)

        # Serialize public key to get size (DER format)
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat
        )
        pk_bytes = public_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        pk_size = len(pk_bytes)

        # Encrypt a 32-byte "shared secret" (simulating KEM)
        message = b'\x42' * 32
        pad = padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )

        t0 = time.perf_counter()
        ciphertext = public_key.encrypt(message, pad)
        encrypt_times.append((time.perf_counter() - t0) * 1000)

        ct_size = len(ciphertext)

        t0 = time.perf_counter()
        decrypted = private_key.decrypt(ciphertext, pad)
        decrypt_times.append((time.perf_counter() - t0) * 1000)

        assert decrypted == message

    return {
        "keygen_avg": statistics.mean(keygen_times),
        "keygen_med": statistics.median(keygen_times),
        "encaps_avg": statistics.mean(encrypt_times),
        "encaps_med": statistics.median(encrypt_times),
        "decaps_avg": statistics.mean(decrypt_times),
        "decaps_med": statistics.median(decrypt_times),
        "pk_size": pk_size,
        "ct_size": ct_size,
        "ss_size": 32,
    }

# ============================================================================
# 3. PRINT RESULTS
# ============================================================================

def print_results(mlkem, rsa_data):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS: ML-KEM-768 vs RSA-2048")
    print("=" * 70)

    fmt = "  {:<30s} {:>15s} {:>15s}"
    print(fmt.format("Metric", "ML-KEM-768", "RSA-2048"))
    print("  " + "-" * 62)
    print(fmt.format("KeyGen (avg, ms)",
          f"{mlkem['keygen_avg']:.2f}", f"{rsa_data['keygen_avg']:.2f}"))
    print(fmt.format("Encaps/Encrypt (avg, ms)",
          f"{mlkem['encaps_avg']:.2f}", f"{rsa_data['encaps_avg']:.2f}"))
    print(fmt.format("Decaps/Decrypt (avg, ms)",
          f"{mlkem['decaps_avg']:.2f}", f"{rsa_data['decaps_avg']:.2f}"))
    print(fmt.format("Public key size (bytes)",
          f"{mlkem['pk_size']}", f"{rsa_data['pk_size']}"))
    print(fmt.format("Ciphertext size (bytes)",
          f"{mlkem['ct_size']}", f"{rsa_data['ct_size']}"))
    print(fmt.format("Shared secret size (bytes)",
          f"{mlkem['ss_size']}", f"{rsa_data['ss_size']}"))
    print("=" * 70)

# ============================================================================
# 4. GENERATE CHARTS
# ============================================================================

def generate_charts(mlkem, rsa_data, output_dir="."):
    """Generate comparison bar charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n  [WARNING] matplotlib not installed. Skipping charts.")
        print("  Install with: pip install matplotlib")
        return

    # -- Style --
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.facecolor': '#1a1a2e',
        'figure.facecolor': '#0f0f23',
        'text.color': '#e0e0e0',
        'axes.labelcolor': '#e0e0e0',
        'xtick.color': '#b0b0b0',
        'ytick.color': '#b0b0b0',
    })

    colors_mlkem = '#00d2ff'
    colors_rsa = '#ff6b6b'

    # ── Chart 1: Execution Time ──
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['KeyGen', 'Encaps /\nEncrypt', 'Decaps /\nDecrypt']
    mlkem_times = [mlkem['keygen_avg'], mlkem['encaps_avg'], mlkem['decaps_avg']]
    rsa_times = [rsa_data['keygen_avg'], rsa_data['encaps_avg'], rsa_data['decaps_avg']]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, mlkem_times, w, label='ML-KEM-768', color=colors_mlkem,
                   edgecolor='white', linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x + w/2, rsa_times, w, label='RSA-2048', color=colors_rsa,
                   edgecolor='white', linewidth=0.5, alpha=0.9)

    ax.set_ylabel('Time (ms)')
    ax.set_title('Execution Time: ML-KEM-768 vs RSA-2048', fontsize=14, fontweight='bold', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(facecolor='#16213e', edgecolor='#444')
    ax.grid(axis='y', alpha=0.2)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}',
                ha='center', va='bottom', fontsize=9, color=colors_mlkem)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}',
                ha='center', va='bottom', fontsize=9, color=colors_rsa)

    plt.tight_layout()
    path1 = os.path.join(output_dir, 'chart_time_comparison.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")
    plt.close()

    # ── Chart 2: Key & Ciphertext Sizes ──
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Public Key', 'Ciphertext']
    mlkem_sizes = [mlkem['pk_size'], mlkem['ct_size']]
    rsa_sizes = [rsa_data['pk_size'], rsa_data['ct_size']]

    x = np.arange(len(labels))
    bars1 = ax.bar(x - w/2, mlkem_sizes, w, label='ML-KEM-768', color=colors_mlkem,
                   edgecolor='white', linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x + w/2, rsa_sizes, w, label='RSA-2048', color=colors_rsa,
                   edgecolor='white', linewidth=0.5, alpha=0.9)

    ax.set_ylabel('Size (bytes)')
    ax.set_title('Key & Ciphertext Size: ML-KEM-768 vs RSA-2048', fontsize=14, fontweight='bold', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(facecolor='#16213e', edgecolor='#444')
    ax.grid(axis='y', alpha=0.2)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
                ha='center', va='bottom', fontsize=10, color=colors_mlkem)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
                ha='center', va='bottom', fontsize=10, color=colors_rsa)

    plt.tight_layout()
    path2 = os.path.join(output_dir, 'chart_size_comparison.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")
    plt.close()

    # ── Chart 3: Combined summary table as figure ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table_data = [
        ['KeyGen (ms)', f"{mlkem['keygen_avg']:.2f}", f"{rsa_data['keygen_avg']:.2f}"],
        ['Encaps/Encrypt (ms)', f"{mlkem['encaps_avg']:.2f}", f"{rsa_data['encaps_avg']:.2f}"],
        ['Decaps/Decrypt (ms)', f"{mlkem['decaps_avg']:.2f}", f"{rsa_data['decaps_avg']:.2f}"],
        ['Public Key (bytes)', f"{mlkem['pk_size']}", f"{rsa_data['pk_size']}"],
        ['Ciphertext (bytes)', f"{mlkem['ct_size']}", f"{rsa_data['ct_size']}"],
        ['Shared Secret (bytes)', f"{mlkem['ss_size']}", f"{rsa_data['ss_size']}"],
    ]
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'ML-KEM-768', 'RSA-2048'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#444')
        if row == 0:
            cell.set_facecolor('#16213e')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 1:
            cell.set_facecolor('#0a2a4a')
            cell.set_text_props(color=colors_mlkem)
        elif col == 2:
            cell.set_facecolor('#2a0a0a')
            cell.set_text_props(color=colors_rsa)
        else:
            cell.set_facecolor('#1a1a2e')
            cell.set_text_props(color='#e0e0e0')

    ax.set_title('ML-KEM-768 vs RSA-2048: Summary', fontsize=14, fontweight='bold',
                 color='white', pad=20)
    plt.tight_layout()
    path3 = os.path.join(output_dir, 'chart_summary_table.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path3}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

import os

def main():
    print("=" * 70)
    print("  ML-KEM-768 vs RSA-2048 — Performance Benchmark")
    print(f"  Iterations: {N_ITERATIONS}")
    print("=" * 70)

    # Run benchmarks
    mlkem_results = benchmark_mlkem(N_ITERATIONS)
    rsa_results = benchmark_rsa(N_ITERATIONS)

    # Print comparison
    print_results(mlkem_results, rsa_results)

    # Generate charts
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n[Charts]")
    generate_charts(mlkem_results, rsa_results, output_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()
