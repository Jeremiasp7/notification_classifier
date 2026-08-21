from scripts.loader import load_all, load_test, load_train, load_val

EXPECTED_CLASSES = {
    "Tarefas e Delegações Internas",
    "Financeiro e Custas",
    "Gestão de Documentos",
    "Prazos e Audiências",
    "Andamentos Processuais",
}


def test_load_train_has_expected_columns():
    df = load_train()
    assert {"id", "sentenca", "classe"}.issubset(df.columns)
    assert len(df) > 0


def test_load_val_and_test_not_empty():
    assert len(load_val()) > 0
    assert len(load_test()) > 0


def test_all_splits_share_the_same_classes():
    data = load_all()
    for split_name, df in data.items():
        assert set(df["classe"].unique()) == EXPECTED_CLASSES, split_name


def test_no_missing_sentences():
    for df in load_all().values():
        assert df["sentenca"].str.len().gt(0).all()
