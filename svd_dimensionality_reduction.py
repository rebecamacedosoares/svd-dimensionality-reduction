import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Valores de k utilizados
K_VALUES = [5, 20, 50, 100, 200]

# Função para análise de uma imagem
def analisar_imagem(arquivo, nome):
    """
    Aplica a SVD a uma imagem em escala de cinza,
    gera as figuras e calcula as métricas utilizadas
    na análise.
    """

    print("\n" + "=" * 70)
    print(f"ANÁLISE DA IMAGEM: {nome.upper()}")
    print("=" * 70)

    # Carregar imagem
    imagem = Image.open(arquivo).convert("L")

    A = np.array(imagem, dtype=float)

    m, n = A.shape

    print(f"Dimensões: {m} x {n}")
    print(f"Total de pixels: {m*n}")

    # Decomposição SVD
    U, S, VT = np.linalg.svd(A, full_matrices=False)

    print(f"Número de valores singulares: {len(S)}")

    ks = [k for k in K_VALUES if k <= min(m, n)]

    energia_total = np.sum(S**2)

    # Figura - imagem original
    plt.figure(figsize=(5, 5))

    plt.imshow(A, cmap="gray")

    plt.title(f"Imagem original ({nome.capitalize()})")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(f"{nome}_original.png", dpi=300)

    plt.show()
    plt.close()

    # Figura - valores singulares
    plt.figure(figsize=(8, 5))

    plt.plot(S)

    plt.title(f"Valores singulares ({nome.capitalize()})")

    plt.xlabel("Índice")

    plt.ylabel("Magnitude")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"{nome}_valores_singulares.png", dpi=300)

    plt.show()
    plt.close()

    # Figura - energia acumulada
    energia_acumulada = np.cumsum(S**2)
    energia_acumulada /= energia_acumulada[-1]

    plt.figure(figsize=(8, 5))

    plt.plot(energia_acumulada)

    plt.title(f"Energia acumulada ({nome.capitalize()})")

    plt.xlabel("k")

    plt.ylabel("Energia preservada")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"{nome}_energia_acumulada.png", dpi=300)

    plt.show()
    plt.close()

    # Reconstruções
    reconstrucoes = {}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(A, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    for ax, k in zip(axes.flat[1:], ks):

        Ak = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

        Ak = np.clip(Ak, 0, 255)

        reconstrucoes[k] = Ak

        ax.imshow(Ak.astype(np.uint8), cmap="gray")

        ax.set_title(f"k = {k}")

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"{nome}_comparacao.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # Resultados numéricos
    print("\nMétricas obtidas:\n")

    print(
        f"{'k':<10}"
        f"{'Erro':<18}"
        f"{'Compressão':<15}"
        f"{'Energia (%)'}"
    )

    resultados = []

    for k in ks:

        Ak = reconstrucoes[k]

        erro = (
            np.linalg.norm(A - Ak, "fro")
            / np.linalg.norm(A, "fro")
        )

        energia = (
            100 * np.sum(S[:k]**2)
            / energia_total
        )

        taxa = (m * n) / (k * (m + n + 1))

        resultados.append(
            (k, erro, taxa, energia)
        )

        print(
            f"{k:<10}"
            f"{erro:<18.6f}"
            f"{taxa:<15.2f}"
            f"{energia:.2f}"
        )

    return S, energia_acumulada, resultados



# Análise das imagens
S_paisagem, energia_paisagem, resultados_paisagem = analisar_imagem(
    "imagem_paisagem.jpg",
    "paisagem"
)

S_folhas, energia_folhas, resultados_folhas = analisar_imagem(
    "imagem_folhas.jpg",
    "folhas"
)


# Comparação dos valores singulares
plt.figure(figsize=(8, 5))

plt.plot(S_paisagem, label="Paisagem")

plt.plot(S_folhas, label="Folhas")

plt.xlabel("Índice")

plt.ylabel("Valor singular")

plt.title("Comparação dos valores singulares")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "comparacao_valores_singulares.png",
    dpi=300
)

plt.show()
plt.close()


# Comparação da energia acumulada
plt.figure(figsize=(8, 5))

plt.plot(energia_paisagem, label="Paisagem")

plt.plot(energia_folhas, label="Folhas")

plt.xlabel("k")

plt.ylabel("Energia acumulada")

plt.title("Comparação da energia acumulada")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "comparacao_energia_acumulada.png",
    dpi=300
)

plt.show()
plt.close()