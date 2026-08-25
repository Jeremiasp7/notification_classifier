import os

import pandas as pd


def gerar_dataset_csv(pasta_alvo='data/sentencas'):
    caminho_saida = os.path.join(pasta_alvo, 'dataset_consolidado.csv')
    dataset = []
    id_contador = 1

    if not os.path.exists(pasta_alvo):
        print(f"A pasta '{pasta_alvo}' não foi encontrada.")
        return None

    arquivos = [f for f in os.listdir(pasta_alvo) if f.endswith('.txt')]

    if not arquivos:
        print(f"Nenhum arquivo .txt encontrado na pasta '{pasta_alvo}'.")
        return None

    for arquivo in arquivos:
        classe = os.path.splitext(arquivo)[0]
        caminho_arquivo = os.path.join(pasta_alvo, arquivo)

        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            for linha in f:
                sentenca = linha.strip()
                if sentenca:
                    dataset.append(
                        {
                            'id': id_contador,
                            'sentenca': sentenca,
                            'classe': classe
                        }
                    )
                    id_contador += 1

    df = pd.DataFrame(dataset)
    df.to_csv(caminho_saida, index=False, encoding='utf-8', sep=';')
    print(f"Dataset gerado com sucesso em: {caminho_saida}")
    return caminho_saida


if __name__ == '__main__':
    gerar_dataset_csv('data/sentencas')
