import streamlit as st
import pandas as pd
import string
import random
from math import gcd
import numpy as np

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

def matrix_mod_inverse(mat, mod=26):
    det = int(np.round(np.linalg.det(mat))) % mod
    det_inv = mod_inverse(det, mod)
    adj = np.linalg.inv(mat) * np.linalg.det(mat)
    inv = (det_inv * adj) % mod
    return inv.astype(int)

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
    return caesar_encrypt(cipher, -shift % 26)

def caesar_steps(ciphertext, shift, mode="decrypt"):
    rows = []
    text_upper = ciphertext.upper()
    
    # Choose operation: subtract for decrypt, add for encrypt
    operation = lambda x, y: (x - y) % 26 if mode == "decrypt" else (x + y) % 26
    start_col = "Cipher" if mode == "decrypt" else "Plain"
    end_col = "Plain" if mode == "decrypt" else "Cipher"
    label = "y - shift" if mode == "decrypt" else "x + shift"
    
    for ch in text_upper:
        if ch.isalpha():
            idx = A2I[ch]
            shifted = operation(idx, shift)
            res_ch = I2A[shifted]
            rows.append({
                start_col: ch,
                f"{start_col.lower()[0]} (index)": idx,
                label: idx - shift if mode == "decrypt" else idx + shift,
                f"({label}) mod 26": shifted,
                end_col: res_ch
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

# ====================== VIGENÈRE CIPHER ======================
def generate_vigenere_key(text, key):
    key = key.upper()
    full_key = (key * (len(text) // len(key) + 1))[:len(text)]
    return full_key

def vigenere_encrypt(plaintext, key):
    plaintext = plaintext.upper()
    full_key = generate_vigenere_key(plaintext, key)
    out, rows = [], []
    for p, k in zip(plaintext, full_key):
        if p.isalpha():
            shift = A2I[k]
            c = I2A[(A2I[p] + shift) % 26]
            out.append(c)
            rows.append({"Plain": p, "Key": k, "Shift": shift, "p + shift mod 26": (A2I[p] + shift) % 26, "Cipher": c})
        else:
            out.append(p)
            rows.append({"Plain": p, "Key": "-", "Shift": "-", "p + shift mod 26": "-", "Cipher": p})
    return "".join(out), pd.DataFrame(rows)

def vigenere_decrypt(ciphertext, key):
    ciphertext = ciphertext.upper()
    full_key = generate_vigenere_key(ciphertext, key)
    out, rows = [], []
    for c, k in zip(ciphertext, full_key):
        if c.isalpha():
            shift = A2I[k]
            p = I2A[(A2I[c] - shift) % 26]
            out.append(p)
            rows.append({"Cipher": c, "Key": k, "Shift": shift, "c - shift mod 26": (A2I[c] - shift) % 26, "Plain": p})
        else:
            out.append(c)
            rows.append({"Cipher": c, "Key": "-", "Shift": "-", "c - shift mod 26": "-", "Plain": c})
    return "".join(out), pd.DataFrame(rows)

# ====================== RAIL FENCE CIPHER ======================
def rail_fence_encrypt(text, rails):
    text = ''.join(c for c in text.upper() if c.isalpha())
    fence = [[''] * len(text) for _ in range(rails)]
    dir_down, row = True, 0
    for i, c in enumerate(text):
        fence[row][i] = c
        if dir_down:
            row += 1
            if row == rails - 1:
                dir_down = False
        else:
            row -= 1
            if row == 0:
                dir_down = True
    cipher = ''.join(c for row in fence for c in row if c)
    steps_df = pd.DataFrame(fence, index=[f"Rail {i+1}" for i in range(rails)]).T
    return cipher, steps_df

def rail_fence_decrypt(cipher, rails):
    cipher = ''.join(c for c in cipher.upper() if c.isalpha())
    fence = [[''] * len(cipher) for _ in range(rails)]
    positions = []
    dir_down, row = True, 0
    for i in range(len(cipher)):
        positions.append((row, i))
        if dir_down:
            row += 1
            if row == rails - 1:
                dir_down = False
        else:
            row -= 1
            if row == 0:
                dir_down = True
    idx = 0
    for r in range(rails):
        for pos in [p for p in positions if p[0] == r]:
            fence[r][pos[1]] = cipher[idx]
            idx += 1
    plain = ''
    dir_down, row = True, 0
    for i in range(len(cipher)):
        plain += fence[row][i]
        if dir_down:
            row += 1
            if row == rails - 1:
                dir_down = False
        else:
            row -= 1
            if row == 0:
                dir_down = True
    steps_df = pd.DataFrame(fence, index=[f"Rail {i+1}" for i in range(rails)]).T
    return plain, steps_df

# ====================== ROW TRANSPOSITION CIPHER ======================
def row_transposition_encrypt(text, key):
    text = ''.join(c for c in text.upper() if c.isalpha())
    cols = len(key)
    rows_num = -(-len(text) // cols)
    grid = [[''] * cols for _ in range(rows_num)]
    for i, c in enumerate(text):
        grid[i // cols][i % cols] = c
    for _ in range(rows_num * cols - len(text)):
        grid[-1].append('X')  # Padding with X
    key_order = sorted(range(cols), key=lambda k: key[k])
    cipher = ''
    for col in key_order:
        for row in grid:
            if row[col]:
                cipher += row[col]
    steps_df = pd.DataFrame(grid, columns=list(key))
    return cipher, steps_df

def row_transposition_decrypt(cipher, key):
    cipher = ''.join(c for c in cipher.upper() if c.isalpha())
    cols = len(key)
    rows_num = len(cipher) // cols
    key_order = sorted(range(cols), key=lambda k: key[k])
    grid = [[''] * cols for _ in range(rows_num)]
    idx = 0
    for col in key_order:
        for r in range(rows_num):
            grid[r][col] = cipher[idx]
            idx += 1
    plain = ''
    for row in grid:
        plain += ''.join(row)
    plain = plain.rstrip('X')
    steps_df = pd.DataFrame(grid, columns=list(key))
    return plain, steps_df

# ====================== PLAYFAIR CIPHER ======================
def playfair_prepare_key(key):
    key = ''.join(c for c in key.upper() if c.isalpha() and c != 'J').replace('J', 'I')
    matrix = ''
    seen = set()
    for c in key + ALPHA:
        if c not in seen and c != 'J':
            seen.add(c)
            matrix += c
    return [list(matrix[i:i+5]) for i in range(0, 25, 5)]

def find_pos(matrix, ch):
    for r, row in enumerate(matrix):
        if ch in row:
            return r, row.index(ch)
    return None

def playfair_encrypt(text, key):
    matrix = playfair_prepare_key(key)
    text = ''.join(c for c in text.upper() if c.isalpha()).replace('J', 'I')
    if len(text) % 2 != 0:
        text += 'X'
    pairs = [text[i:i+2] for i in range(0, len(text), 2)]
    out, rows = '', []
    for p1, p2 in pairs:
        if p1 == p2:
            p2 = 'X'
        r1, c1 = find_pos(matrix, p1)
        r2, c2 = find_pos(matrix, p2)
        if r1 == r2:
            c1_out, c2_out = (c1 + 1) % 5, (c2 + 1) % 5
            out += matrix[r1][c1_out] + matrix[r2][c2_out]
        elif c1 == c2:
            r1_out, r2_out = (r1 + 1) % 5, (r2 + 1) % 5
            out += matrix[r1_out][c1] + matrix[r2_out][c2]
        else:
            out += matrix[r1][c2] + matrix[r2][c1]
        rows.append({"Pair": f"{p1}{p2}", "Rule": "Same row/col/rect", "Cipher Pair": out[-2:]})
    steps_df = pd.DataFrame(rows)
    return out, steps_df, pd.DataFrame(matrix)

def playfair_decrypt(cipher, key):
    matrix = playfair_prepare_key(key)
    pairs = [cipher[i:i+2] for i in range(0, len(cipher), 2)]
    out, rows = '', []
    for c1, c2 in pairs:
        r1, col1 = find_pos(matrix, c1)
        r2, col2 = find_pos(matrix, c2)
        if r1 == r2:
            col1_out, col2_out = (col1 - 1) % 5, (col2 - 1) % 5
            out += matrix[r1][col1_out] + matrix[r2][col2_out]
        elif col1 == col2:
            r1_out, r2_out = (r1 - 1) % 5, (r2 - 1) % 5
            out += matrix[r1_out][col1] + matrix[r2_out][col2]
        else:
            out += matrix[r1][col2] + matrix[r2][col1]
        rows.append({"Pair": f"{c1}{c2}", "Rule": "Same row/col/rect", "Plain Pair": out[-2:]})
    steps_df = pd.DataFrame(rows)
    return out.rstrip('X'), steps_df, pd.DataFrame(matrix)

# ====================== HILL CIPHER ======================
def hill_encrypt(text, key_matrix):
    n = key_matrix.shape[0]
    text = ''.join(c for c in text.upper() if c.isalpha())
    if len(text) % n != 0:
        text += 'X' * (n - len(text) % n)
    out = ''
    rows = []
    for i in range(0, len(text), n):
        vec = np.array([A2I[c] for c in text[i:i+n]])
        res = (key_matrix @ vec) % 26
        cipher_chunk = ''.join(I2A[int(x)] for x in res)
        out += cipher_chunk
        rows.append({"Plain Vector": vec.tolist(), "After Multiply": (key_matrix @ vec).tolist(), "Mod 26": res.tolist(), "Cipher": cipher_chunk})
    return out, pd.DataFrame(rows)

def hill_decrypt(cipher, key_matrix):
    n = key_matrix.shape[0]
    cipher = ''.join(c for c in cipher.upper() if c.isalpha())
    inv_key = matrix_mod_inverse(key_matrix)
    out = ''
    rows = []
    for i in range(0, len(cipher), n):
        vec = np.array([A2I[c] for c in cipher[i:i+n]])
        res = (inv_key @ vec) % 26
        plain_chunk = ''.join(I2A[int(x)] for x in res)
        out += plain_chunk
        rows.append({"Cipher Vector": vec.tolist(), "After Multiply": (inv_key @ vec).tolist(), "Mod 26": res.tolist(), "Plain": plain_chunk})
    return out.rstrip('X'), pd.DataFrame(rows)

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
st.title("🔐 Cryptographic Algorithms")
st.markdown("Explore classical ciphers with encryption, decryption, and step-by-step breakdowns.")

tabs = st.tabs(["Caesar", "Affine", "Vigenère", "Rail Fence", "Row Transposition", "Playfair", "Hill", "Rotor"])

# ------------------- TAB 1: CAESAR -------------------
with tabs[0]:
    st.header("Caesar Cipher")
    caesar_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="caesar_mode")
    col1, col2 = st.columns([3, 1])
    with col1:
        caesar_text = st.text_area("Text", "Hasnaat", height=100, key="caesar_text")
    with col2:
        caesar_shift = st.slider("Shift (1–25)", 1, 25, 3, key="caesar_shift")

    if st.button("Run", key="run_caesar"):
        if not caesar_text.strip():
            st.warning("Enter text.")
        else:
            if caesar_mode == "Encrypt":
                result = caesar_encrypt(caesar_text, caesar_shift)
                st.success(f"**Ciphertext:** `{result}`")
                steps = caesar_steps(caesar_text, caesar_shift, mode="encrypt")
            else:
                result = caesar_decrypt(caesar_text, caesar_shift)
                st.success(f"**Plaintext:** `{result}`")
                steps = caesar_steps(caesar_text, caesar_shift, mode="decrypt")
            if not steps.empty:
                st.markdown("#### Steps")
                st.dataframe(steps, use_container_width=True)

# ------------------- TAB 2: AFFINE -------------------
with tabs[1]:
    st.header("Affine Cipher")
    affine_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="affine_mode")
    col1, col2 = st.columns(2)
    with col1:
        affine_msg = st.text_area("Message", "Hasnaat", height=100, key="affine_msg")
    with col2:
        affine_a = st.number_input("Key 'a' (coprime with 26)", min_value=1, value=3, key="affine_a")
        affine_b = st.number_input("Key 'b'", min_value=0, value=5, key="affine_b")

    if st.button("Run", key="run_affine"):
        if not affine_msg.strip():
            st.warning("Enter message.")
        elif gcd(affine_a, 26) != 1:
            st.error(f"`a={affine_a}` not coprime with 26.")
        else:
            if affine_mode == "Encrypt":
                result, steps = affine_encrypt(affine_msg, affine_a, affine_b)
                st.success(f"**Ciphertext:** `{result}`")
            else:
                result, steps = affine_decrypt(affine_msg, affine_a, affine_b)
                st.success(f"**Plaintext:** `{result}`")
            st.markdown("#### Steps")
            st.dataframe(steps, use_container_width=True)

# ------------------- TAB 3: VIGENÈRE -------------------
with tabs[2]:
    st.header("Vigenère Cipher")
    vigenere_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="vigenere_mode")
    col1, col2 = st.columns([3, 1])
    with col1:
        vigenere_text = st.text_area("Text", "Hasnaat", height=100, key="vigenere_text")
    with col2:
        vigenere_key = st.text_input("Key (word)", "KEY", key="vigenere_key")

    if st.button("Run", key="run_vigenere"):
        if not vigenere_text.strip():
            st.warning("Enter text.")
        elif not vigenere_key.strip():
            st.warning("Enter key.")
        else:
            if vigenere_mode == "Encrypt":
                result, steps = vigenere_encrypt(vigenere_text, vigenere_key)
                st.success(f"**Ciphertext:** `{result}`")
            else:
                result, steps = vigenere_decrypt(vigenere_text, vigenere_key)
                st.success(f"**Plaintext:** `{result}`")
            st.markdown("#### Steps")
            st.dataframe(steps, use_container_width=True)

# ------------------- TAB 4: RAIL FENCE -------------------
with tabs[3]:
    st.header("Rail Fence Cipher")
    rail_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="rail_mode")
    col1, col2 = st.columns([3, 1])
    with col1:
        rail_text = st.text_area("Text", "Hasnaat", height=100, key="rail_text")
    with col2:
        rail_rails = st.number_input("Rails", min_value=2, value=3, key="rail_rails")

    if st.button("Run", key="run_rail"):
        if not rail_text.strip():
            st.warning("Enter text.")
        else:
            if rail_mode == "Encrypt":
                result, steps = rail_fence_encrypt(rail_text, rail_rails)
                st.success(f"**Ciphertext:** `{result}`")
            else:
                result, steps = rail_fence_decrypt(rail_text, rail_rails)
                st.success(f"**Plaintext:** `{result}`")
            st.markdown("#### Rail Grid")
            st.dataframe(steps, use_container_width=True)

# ------------------- TAB 5: ROW TRANSPOSITION -------------------
with tabs[4]:
    st.header("Row Transposition Cipher")
    row_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="row_mode")
    col1, col2 = st.columns([3, 1])
    with col1:
        row_text = st.text_area("Text", "Hasnaat", height=100, key="row_text")
    with col2:
        row_key = st.text_input("Key (word)", "KEY", key="row_key")

    if st.button("Run", key="run_row"):
        if not row_text.strip():
            st.warning("Enter text.")
        elif not row_key.strip() or len(set(row_key)) != len(row_key):
            st.warning("Enter unique key letters.")
        else:
            if row_mode == "Encrypt":
                result, steps = row_transposition_encrypt(row_text, row_key.upper())
                st.success(f"**Ciphertext:** `{result}`")
            else:
                result, steps = row_transposition_decrypt(row_text, row_key.upper())
                st.success(f"**Plaintext:** `{result}`")
            st.markdown("#### Grid (Columns by Key)")
            st.dataframe(steps, use_container_width=True)

# ------------------- TAB 6: PLAYFAIR -------------------
with tabs[5]:
    st.header("Playfair Cipher")
    playfair_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="playfair_mode")
    col1, col2 = st.columns([3, 1])
    with col1:
        playfair_text = st.text_area("Text", "Hasnaat", height=100, key="playfair_text")
    with col2:
        playfair_key = st.text_input("Key (word)", "PLAYFAIR", key="playfair_key")

    if st.button("Run", key="run_playfair"):
        if not playfair_text.strip():
            st.warning("Enter text.")
        else:
            if playfair_mode == "Encrypt":
                result, steps, matrix_df = playfair_encrypt(playfair_text, playfair_key)
                st.success(f"**Ciphertext:** `{result}`")
            else:
                result, steps, matrix_df = playfair_decrypt(playfair_text, playfair_key)
                st.success(f"**Plaintext:** `{result}`")
            st.markdown("#### Matrix")
            st.dataframe(matrix_df, use_container_width=True)
            st.markdown("#### Steps")
            st.dataframe(steps, use_container_width=True)

# ------------------- TAB 7: HILL -------------------
with tabs[6]:
    st.header("Hill Cipher")
    hill_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="hill_mode")
    hill_text = st.text_area("Text", "PAYMOREMONEY", height=100, key="hill_text")
    hill_n = st.number_input("Matrix Size (n x n)", min_value=2, max_value=4, value=2, key="hill_n")
    st.markdown("Enter Key Matrix (row-wise, space-separated integers)")
    hill_key_str = st.text_input("Key Matrix", "6 24 1 13", key="hill_key")  # Example for 2x2
    if st.button("Run", key="run_hill"):
        if not hill_text.strip():
            st.warning("Enter text.")
        else:
            try:
                key_list = list(map(int, hill_key_str.split()))
                if len(key_list) != hill_n ** 2:
                    raise ValueError(f"Enter {hill_n**2} numbers.")
                key_matrix = np.array(key_list).reshape(hill_n, hill_n)
                if hill_mode == "Encrypt":
                    result, steps = hill_encrypt(hill_text, key_matrix)
                    st.success(f"**Ciphertext:** `{result}`")
                else:
                    result, steps = hill_decrypt(hill_text, key_matrix)
                    st.success(f"**Plaintext:** `{result}`")
                st.markdown("#### Key Matrix")
                st.dataframe(pd.DataFrame(key_matrix), use_container_width=True)
                st.markdown("#### Steps")
                st.dataframe(steps, use_container_width=True)
            except Exception as e:
                st.error(str(e))

# ------------------- TAB 8: ROTOR -------------------
with tabs[7]:
    st.header("Rotor Machine (Enigma Simulation)")
    rotor_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="rotor_mode")
    col1, col2 = st.columns([3, 1])
    with col1:
        rotor_text = st.text_area("Text", "ATIFASLAM", height=100, key="rotor_text")
    with col2:
        num_rotors = st.number_input("Rotors", min_value=1, max_value=4, value=2, key="num_rotors")

    if st.button("Run", key="run_rotor"):
        if not rotor_text.strip():
            st.warning("Enter text.")
        else:
            rotors = [make_rotor(seed=i+1) for i in range(num_rotors)]
            with st.expander("Rotor Wiring", expanded=False):
                for i, r in enumerate(rotors, 1):
                    st.write(f"**Rotor {i}:** `{r}`")
            if rotor_mode == "Encrypt":
                result, steps, final_pos = rotor_encrypt(rotor_text, rotors)
                st.success(f"**Ciphertext:** `{result}`")
            else:
                result, steps, final_pos = rotor_decrypt(rotor_text, rotors)
                st.success(f"**Plaintext:** `{result}`")
            st.markdown("#### Steps Log")
            st.dataframe(steps, use_container_width=True)
            st.caption(f"Final Positions: {final_pos}")

# ====================== FOOTER ======================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Classical Ciphers Collection")