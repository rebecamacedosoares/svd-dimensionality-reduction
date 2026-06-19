import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Carregar imagem colorida
imagem_colorida = Image.open("imagem.jpg")

# Mostrar imagem colorida - apenas para referência
plt.figure(figsize=(6, 6))

plt.imshow(imagem_colorida)
plt.title("Imagem Original Colorida")
plt.axis("off")

plt.show()

# Carregar imagem e converter para escala de cinza
imagem = Image.open("imagem.jpg").convert("L")
A = np.array(imagem, dtype=float)

m, n = A.shape

print("=" * 60)
print("INFORMAÇÕES DA IMAGEM")
print("=" * 60)
print(f"Dimensões: {m} x {n}")
print(f"Total de pixels: {m*n}")

# Aplicar SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

print(f"Número de valores singulares: {len(S)}")

# Valores de k
ks = [5, 20, 50, 100, 200]

# Garantir que k não ultrapasse o limite da imagem
ks = [k for k in ks if k <= min(m, n)]

# Imagem original
plt.figure(figsize=(6, 6))

plt.imshow(A, cmap="gray")
plt.title("Imagem Original")
plt.axis("off")

plt.tight_layout()
plt.savefig("imagem_original.png", dpi=300)

plt.show()

# Valores singulares
plt.figure(figsize=(8, 5))

plt.plot(S)

plt.title("Valores Singulares")
plt.xlabel("Índice")
plt.ylabel("Magnitude")

plt.grid(True)

plt.tight_layout()
plt.savefig("valores_singulares.png", dpi=300)

plt.show()

# Energia acumulada
energia_acumulada = np.cumsum(S**2)
energia_acumulada /= energia_acumulada[-1]

plt.figure(figsize=(8, 5))

plt.plot(energia_acumulada)

plt.title("Energia Acumulada")
plt.xlabel("k")
plt.ylabel("Energia preservada")

plt.grid(True)

plt.tight_layout()
plt.savefig("energia_acumulada.png", dpi=300)

plt.show()

# Energia preservada
energia_total = np.sum(S**2)

print("\n")
print("=" * 60)
print("ENERGIA PRESERVADA")
print("=" * 60)

for k in ks:

    energia = np.sum(S[:k]**2)

    percentual = 100 * energia / energia_total

    print(f"k={k:3d} -> {percentual:.2f}%")

# Reconstruções
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes[0, 0].imshow(A, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

for ax, k in zip(axes.flat[1:], ks):

    Ak = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

    Ak = np.clip(Ak, 0, 255)

    ax.imshow(Ak.astype(np.uint8), cmap="gray")

    ax.set_title(f"k = {k}")
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "comparacao_svd.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Tabela de resultados
print("\n")
print("=" * 60)
print("RESULTADOS")
print("=" * 60)

print(
    f"{'k':<10}"
    f"{'Erro Relativo':<20}"
    f"{'Compressão':<15}"
    f"{'Energia (%)'}"
)

for k in ks:

    Ak = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

    erro = (
        np.linalg.norm(A - Ak, "fro")
        / np.linalg.norm(A, "fro")
    )

    energia = (
        100 * np.sum(S[:k]**2)
        / np.sum(S**2)
    )

    original = m * n

    armazenado = k * (m + n + 1)

    taxa = original / armazenado

    print(
        f"{k:<10}"
        f"{erro:<20.6f}"
        f"{taxa:<15.2f}"
        f"{energia:.2f}"
    )

print("\nArquivos gerados:")
print("- imagem_original.png")
print("- valores_singulares.png")
print("- energia_acumulada.png")
print("- comparacao_svd.png")
