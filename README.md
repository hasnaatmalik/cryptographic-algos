# 🔐 CryptoLab - Classical Cryptographic Algorithms

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cryptographic-algos-rr5mktwx7bgusaalmiqctp.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive educational web application for exploring classical cryptographic algorithms. Built with Streamlit, this app provides hands-on experience with encryption and decryption operations, complete with step-by-step breakdowns to enhance understanding.

## 🌐 Live Demo

**[Try the App →](https://cryptographic-algos-rr5mktwx7bgusaalmiqctp.streamlit.app/)**

## ✨ Features

### Supported Ciphers

| Cipher | Description |
|--------|-------------|
| **Caesar Cipher** | A substitution cipher that shifts letters by a fixed amount |
| **Affine Cipher** | Uses modular arithmetic with formula `E(x) = (ax + b) mod 26` |
| **Vigenère Cipher** | Polyalphabetic cipher using a keyword for variable shifts |
| **Rail Fence Cipher** | Transposition cipher that writes message in zigzag pattern |
| **Row Transposition** | Rearranges columns based on a keyword permutation |
| **Playfair Cipher** | Digraph substitution using a 5×5 key matrix |
| **Hill Cipher** | Matrix-based polygraphic cipher using linear algebra |
| **Rotor Machine** | Enigma-style encryption with configurable rotors |

### Key Capabilities

- 🔄 **Encrypt & Decrypt** - Both operations available for all ciphers
- 📊 **Step-by-Step Visualization** - See exactly how each algorithm transforms your text
- 🎛️ **Interactive Controls** - Adjust parameters like shift values, keys, and matrix sizes
- 📈 **Visual Grids & Tables** - Rail fence grids, Playfair matrices, Hill cipher vectors displayed beautifully
- 🎨 **Modern UI** - Clean, tabbed interface for easy navigation between ciphers

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hasnaatmalik/cryptographic-algos.git
   cd cryptographic-algos
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   
   Navigate to `http://localhost:8501` to access the app.

## 📁 Project Structure

```
cryptographic-algos/
├── app.py              # Main Streamlit application with all cipher implementations
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── main.ipynb         # Jupyter notebook with additional experiments
```

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation for step visualization
- **[NumPy](https://numpy.org/)** - Matrix operations for Hill cipher

## 📚 Cipher Details

### Caesar Cipher
A simple substitution cipher where each letter is shifted by a fixed number. For example, with shift=3: A→D, B→E, C→F.

### Affine Cipher
Encrypts using the formula: `C = (aP + b) mod 26`  
Decrypts using: `P = a⁻¹(C - b) mod 26`  
Requires `a` to be coprime with 26.

### Vigenère Cipher
Uses a repeating keyword where each letter of the key determines the shift for the corresponding plaintext letter.

### Rail Fence Cipher
Writes plaintext in a zigzag pattern across multiple "rails" and reads off row by row to create ciphertext.

### Row Transposition
Arranges text in a grid based on keyword length, then reads columns in alphabetical order of the keyword.

### Playfair Cipher
Uses a 5×5 matrix built from a keyword. Encrypts pairs of letters using three rules based on their positions in the matrix.

### Hill Cipher
Uses matrix multiplication with a key matrix. Plaintext vectors are multiplied by the key matrix modulo 26.

### Rotor Machine
Simulates Enigma-style encryption with configurable rotors that advance after each letter, creating polyalphabetic substitution.

## 🎓 Educational Use

This project was developed as part of the **CS3002 - Information Security** course. It serves as a practical tool for understanding:

- How classical encryption algorithms work
- The mathematical foundations of cryptography
- The difference between substitution and transposition ciphers
- Historical encryption methods like Enigma

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Hasnaat Malik**

- GitHub: [@hasnaatmalik](https://github.com/hasnaatmalik)

---

<p align="center">
  Built with ❤️ using Streamlit | Classical Ciphers Collection
</p>
