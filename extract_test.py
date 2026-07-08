import fitz  # PyMuPDF

PDF_PATH = "documents/ThinkBig_Brochure.pdf"

doc = fitz.open(PDF_PATH)

for page_number, page in enumerate(doc, start=1):
    print("=" * 80)
    print(f"PAGE {page_number}")
    print("=" * 80)

    blocks = page.get_text("blocks")

    # Chaque bloc contient notamment : x0, y0, x1, y1, text...
    # On trie les blocs du haut vers le bas, puis de gauche à droite.
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    for block in blocks:
        text = block[4].strip()
        if text:
            text = " ".join(text.split())
            print(text)
            print()