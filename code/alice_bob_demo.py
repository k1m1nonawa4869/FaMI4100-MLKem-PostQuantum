"""
Cụm 2: Alice & Bob Communication Simulation
=============================================
Demonstrates ML-KEM-768 key exchange protocol step-by-step.
"""

import time
import sys
import os

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

from code.ml_kem_core import (
    ml_kem_keygen, ml_kem_encaps, ml_kem_decaps, K
)

def print_hex(label, data, max_bytes=32):
    """Pretty-print binary data as hex."""
    hex_str = data.hex()
    if len(data) > max_bytes:
        hex_str = hex_str[:max_bytes*2] + "..."
    print(f"    {label}: {hex_str}")
    print(f"    ({len(data)} bytes)")

def separator():
    print("─" * 60)

def main():
    print()
    print("╔" + "═"*58 + "╗")
    print("║" + " ML-KEM-768 Key Exchange: Alice & Bob Simulation ".center(58) + "║")
    print("║" + " (FIPS 203 / CRYSTALS-Kyber) ".center(58) + "║")
    print("╚" + "═"*58 + "╝")
    print()

    # ── STEP 1: Alice generates keypair ──
    separator()
    print("[STEP 1] Alice: Generating keypair...")
    print("    Running ML-KEM.KeyGen() ...")
    t0 = time.perf_counter()
    ek, dk = ml_kem_keygen()
    t_keygen = (time.perf_counter() - t0) * 1000
    print(f"    Done in {t_keygen:.1f} ms")
    print()
    print_hex("Encapsulation key (public) ", ek)
    print()
    print_hex("Decapsulation key (private)", dk)
    print()
    print(f"    Alice keeps dk SECRET and sends ek to Bob over the public channel.")
    print()

    # ── STEP 2: Alice sends public key to Bob ──
    separator()
    print("[STEP 2] Alice ──[public channel]──> Bob")
    print(f"    Transmitting encapsulation key: {len(ek)} bytes")
    print(f"    (This key can be intercepted — it is PUBLIC)")
    print()

    # ── STEP 3: Bob encapsulates ──
    separator()
    print("[STEP 3] Bob: Encapsulating shared secret...")
    print("    Running ML-KEM.Encaps(ek) ...")
    t0 = time.perf_counter()
    shared_key_bob, ciphertext = ml_kem_encaps(ek)
    t_encaps = (time.perf_counter() - t0) * 1000
    print(f"    Done in {t_encaps:.1f} ms")
    print()
    print_hex("Bob's shared secret key    ", shared_key_bob)
    print()
    print_hex("Ciphertext                 ", ciphertext)
    print()
    print(f"    Bob keeps the shared key SECRET and sends the ciphertext to Alice.")
    print()

    # ── STEP 4: Bob sends ciphertext to Alice ──
    separator()
    print("[STEP 4] Bob ──[public channel]──> Alice")
    print(f"    Transmitting ciphertext: {len(ciphertext)} bytes")
    print(f"    (An eavesdropper sees this but CANNOT extract the key)")
    print()

    # ── STEP 5: Alice decapsulates ──
    separator()
    print("[STEP 5] Alice: Decapsulating shared secret...")
    print("    Running ML-KEM.Decaps(dk, ciphertext) ...")
    print("    (Includes Fujisaki-Okamoto re-encryption check)")
    t0 = time.perf_counter()
    shared_key_alice = ml_kem_decaps(dk, ciphertext)
    t_decaps = (time.perf_counter() - t0) * 1000
    print(f"    Done in {t_decaps:.1f} ms")
    print()
    print_hex("Alice's shared secret key  ", shared_key_alice)
    print()

    # ── STEP 6: Verification ──
    separator()
    print("[STEP 6] Verification")
    print()
    print(f"    Bob's key:   {shared_key_bob.hex()}")
    print(f"    Alice's key: {shared_key_alice.hex()}")
    print()

    if shared_key_alice == shared_key_bob:
        print("    [OK] Keys MATCH! Secure shared secret established.")
        print()
        print("    Both Alice and Bob now share the same 256-bit key.")
        print("    They can use it with AES-256-GCM to encrypt messages.")
    else:
        print("    [FAIL] Keys DO NOT match!")
    print()

    # ── Summary ──
    separator()
    print("[Summary]")
    print(f"    Parameter set:        ML-KEM-768 (k={K})")
    print(f"    Public key size:      {len(ek)} bytes")
    print(f"    Ciphertext size:      {len(ciphertext)} bytes")
    print(f"    Shared secret size:   {len(shared_key_bob)} bytes (256 bits)")
    print(f"    KeyGen time:          {t_keygen:.1f} ms")
    print(f"    Encaps time:          {t_encaps:.1f} ms")
    print(f"    Decaps time:          {t_decaps:.1f} ms")
    separator()

if __name__ == "__main__":
    main()
