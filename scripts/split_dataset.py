import os

import pandas as pd
from sklearn.model_selection import train_test_split


def gerar_splits(
        arquivo_entrada='data/sentencas/dataset_consolidado.csv',
        pasta_saida='data'):
    """
    Lê o dataset consolidado, embaralha e divide em Treino (70%),
    Validação (15%) e Teste (15%) de forma estratificada (balanceada).
    """
    if not os.path.exists(arquivo_entrada):
        print(f"Erro: O arquivo '{arquivo_entrada}' não foi encontrado.")
        return

    df = pd.read_csv(arquivo_entrada, sep=';')

    df_train, df_temp = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df['classe'],
        shuffle=True
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=42,
        stratify=df_temp['classe'],
        shuffle=True
    )

    os.makedirs(pasta_saida, exist_ok=True)

    caminho_train = os.path.join(pasta_saida, 'train.csv')
    caminho_val = os.path.join(pasta_saida, 'val.csv')
    caminho_test = os.path.join(pasta_saida, 'test.csv')

    df_train.to_csv(caminho_train, index=False, encoding='utf-8', sep=';')
    df_val.to_csv(caminho_val, index=False, encoding='utf-8', sep=';')
    df_test.to_csv(caminho_test, index=False, encoding='utf-8', sep=';')

    print("Separação concluída com sucesso!")
    print(f"Treino (70%): {len(df_train)} registros salvos em {caminho_train}")
    print(f"Validação (15%): {len(df_val)} registros salvos em {caminho_val}")
    print(f"Teste (15%): {len(df_test)} registros salvos em {caminho_test}")


if __name__ == '__main__':
    gerar_splits()
