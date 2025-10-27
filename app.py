import streamlit as st
import pandas as pd
import string
import random
from math import gcd

# ====================== CONFIG & HELPERS ======================
st.set_page_config(page_title="CryptoLab", page_icon="🔐", layout="wide")

ALPHA = string.ascii_uppercase
A2I = {c: i for i, c in enumerate(ALPHA)}
I2A = {i: c for i, c in enumerate(ALPHA)}

def mod_inverse(a: int, m: int = 26) -> int:
    try:
        return pow(a, -1, m)
    except ValueError:
        raise ValueError(f"No modular inverse for a={a} under mod {m}. Must be coprime with 26.")

# ====================== CAESAR CIPHER ======================
def caesar_encrypt(text, shift):
    result = ""
    for char in text:
        if char.isupper():
            result += chr((ord(char) - 65 + shift) % 26 + 65)
        elif char.islower():
            result += chr((ord(char) - 97 + shift) % 26 + 97)
        else:
            result += char
    return result

def caesar_decrypt(cipher, shift):
    return caesar_encrypt(cipher, -shift)

def caesar_steps(ciphertext, shift):
    rows = []
    cipher_upper = ciphertext.upper()
    for ch in cipher_upper:
        if ch.isalpha():
            y = ord(ch) - 65
            y_minus_k = (y - shift) % 26
            plain = chr(y_minus_k + 65)
            rows.append({
                "Cipher": ch,
                "y (index)": y,
                "y - shift": y - shift,
                "(y - shift) mod 26": y_minus_k,
                "Plain": plain
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ====================== AFFINE CIPHER ======================
def affine_encrypt(plaintext: str, a: int, b: int):
    out, rows = [], []
    for ch in plaintext.upper():
        if ch.isalpha():
            x = A2I[ch]
            y = (a * x + b) % 26
            c = I2A[y]
            out.append(c)
            rows.append({"Plain": ch, "x": x, "a×x + b": a*x + b, "(a×x + b) mod 26": y, "Cipher": c})
        else:
            out.append(ch)
            rows.append({"Plain": ch, "x": "-", "a×x + b": "-", "(a×x + b) mod 26": "-", "Cipher": ch})
    return "".join(out), pd.DataFrame(rows)

def affine_decrypt(ciphertext: str, a: int, b: int):
    a_inv = mod_inverse(a)
    out, rows = [], []
    for ch in ciphertext.upper():
        if ch.isalpha():
            y = A2I[ch]
            x = (a_inv * (y - b)) % 26
            p = I2A[x]
            out.append(p)
            rows.append({
                "Cipher": ch, "y": y, "y - b": y - b,
                f"a⁻¹ × (y - b)": a_inv * (y - b),
                f"a⁻¹ × (y - b) mod 26": x, "Plain": p
            })
        else:
            out.append(ch)
            rows.append({"Cipher": ch, "y": "-", "y - b": "-", "a⁻¹ × (y - b)": "-", "a⁻¹ × (y - b) mod 26": "-", "Plain": ch})
    return "".join(out), pd.DataFrame(rows)

# ====================== ROTOR CIPHER ======================
def make_rotor(seed=None):
    random.seed(seed)
    perm = list(ALPHA)
    random.shuffle(perm)
    return ''.join(perm)

def rotate_rotor(rotor, n):
    n %= 26
    return rotor[n:] + rotor[:n]

def forward(rotor, letter):
    return rotor[A2I[letter]]

def backward(rotor, letter):
    return I2A[rotor.index(letter)]

def rotor_encrypt(text, rotors, initial_pos=None):
    text = ''.join(c for c in text.upper() if c.isalpha())
    pos = initial_pos[:] if initial_pos else [0] * len(rotors)
    result, steps = "", []
    for i, ch in enumerate(text):
        current = ch
        current_rotors = [rotate_rotor(rotors[j], pos[j]) for j in range(len(rotors))]
        for r in current_rotors:
            current = forward(r, current)
        result += current
        steps.append({"Pos": i+1, "Input": ch, "Output": current, "Rotor Pos": pos.copy()})
        pos[0] += 1
        for j in range(len(pos)-1):
            if pos[j] == 26:
                pos[j] = 0
                pos[j+1] += 1
    return result, pd.DataFrame(steps), pos

def rotor_decrypt(cipher, rotors, initial_pos=None):
    cipher = ''.join(c for c in cipher.upper() if c.isalpha())
    pos = initial_pos[:] if initial_pos else [0] * len(rotors)
    result, steps = "", []
    for i, ch in enumerate(cipher):
        current = ch
        current_rotors = [rotate_rotor(rotors[j], pos[j]) for j in range(len(rotors))]
        for r in reversed(current_rotors):
            current = backward(r, current)
        result += current
        steps.append({"Pos": i+1, "Input": ch, "Output": current, "Rotor Pos": pos.copy()})
        pos[0] += 1
        for j in range(len(pos)-1):
            if pos[j] == 26:
                pos[j] = 0
                pos[j+1] += 1
    return result, pd.DataFrame(steps), pos

# ====================== STREAMLIT UI ======================
st.title("🔐 Cryptographic Algorithms Explorer")
st.markdown("Explore **Caesar**, **Affine**, and **Rotor (Enigma-style)** ciphers with full step-by-step breakdowns.")

tab1, tab2, tab3 = st.tabs(["Caesar Cipher", "Affine Cipher", "Rotor Cipher"])

# ------------------- TAB 1: CAESAR -------------------
with tab1:
    st.header("Caesar Cipher")
    col1, col2 = st.columns([3, 1])
    with col1:
        caesar_text = st.text_area("Enter text", "Hasnaat", height=100, key="caesar_text")
    with col2:
        caesar_shift = st.slider("Shift (1–25)", 1, 25, 3, key="caesar_shift")

    if st.button("Run Caesar Cipher", key="run_caesar"):
        if not caesar_text.strip():
            st.warning("Please enter some text.")
        else:
            encrypted = caesar_encrypt(caesar_text, caesar_shift)
            decrypted = caesar_decrypt(encrypted, caesar_shift)

            st.success(f"**Encrypted:** `{encrypted}`")
            st.info(f"**Decrypted:** `{decrypted}`")

            steps_df = caesar_steps(encrypted, caesar_shift)
            if not steps_df.empty:
                st.markdown("#### Step-by-Step Decryption")
                st.dataframe(steps_df, use_container_width=True)

# ------------------- TAB 2: AFFINE -------------------
with tab2:
    st.header("Affine Cipher")
    affine_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="affine_mode")
    col1, col2 = st.columns(2)
    with col1:
        affine_msg = st.text_area("Message", "Hasnaat", height=100, key="affine_msg")
    with col2:
        affine_a = st.number_input("Key 'a' (coprime with 26)", min_value=1, value=3, key="affine_a")
        affine_b = st.number_input("Key 'b'", min_value=0, value=5, key="affine_b")

    if st.button("Run Affine Cipher", key="run_affine"):
        if not affine_msg.strip():
            st.warning("Please enter a message.")
        elif gcd(affine_a, 26) != 1:
            st.error(f"Error: `a={affine_a}` is **not coprime** with 26. GCD = {gcd(affine_a, 26)}. Choose from: 1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25.")
        else:
            if affine_mode == "Encrypt":
                cipher, steps = affine_encrypt(affine_msg, affine_a, affine_b)
                st.success(f"**Ciphertext:** `{cipher}`")
                st.markdown("#### Encryption Steps")
                st.dataframe(steps, use_container_width=True)
            else:
                try:
                    plain, steps = affine_decrypt(affine_msg, affine_a, affine_b)
                    st.success(f"**Recovered Plaintext:** `{plain}`")
                    st.markdown("#### Decryption Steps")
                    st.dataframe(steps, use_container_width=True)
                except Exception as e:
                    st.error(str(e))

# ------------------- TAB 3: ROTOR -------------------
with tab3:
    st.header("Rotor Cipher (Enigma Simulation)")
    rotor_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="rotor_mode")
    col1, col2 = st.columns([3, 1])
    with col1:
        rotor_text = st.text_area("Text", "ATIFASLAM", height=100, key="rotor_text")
    with col2:
        num_rotors = st.number_input("Number of Rotors", min_value=1, max_value=4, value=2, key="num_rotors")

    if st.button("Run Rotor Cipher", key="run_rotor"):
        if not rotor_text.strip():
            st.warning("Please enter text.")
        else:
            rotors = [make_rotor(seed=i+1) for i in range(num_rotors)]
            with st.expander("Rotor Wiring (Click to view)", expanded=False):
                for i, r in enumerate(rotors, 1):
                    st.write(f"**Rotor {i}:** `{r}`")

            if rotor_mode == "Encrypt":
                cipher, steps, final_pos = rotor_encrypt(rotor_text, rotors)
                st.success(f"**Ciphertext:** `{cipher}`")
            else:
                plain, steps, final_pos = rotor_decrypt(rotor_text, rotors)
                st.success(f"**Recovered Plaintext:** `{plain}`")

            st.markdown("#### Step-by-Step Rotor Log")
            st.dataframe(steps, use_container_width=True)

            st.caption(f"Final rotor positions: {final_pos}")

# ====================== FOOTER ======================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Inspired by classical cryptography")