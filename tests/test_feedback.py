import csv

from app import feedback


def test_known_classes_come_from_txt_files(tmp_path, monkeypatch):
    sentences_dir = tmp_path / "sentencas"
    sentences_dir.mkdir()
    (sentences_dir / "Classe_B.txt").write_text("exemplo", encoding="utf-8")
    (sentences_dir / "Classe_A.txt").write_text("exemplo", encoding="utf-8")
    monkeypatch.setattr(feedback, "SENTENCES_DIR", sentences_dir)

    assert feedback.known_classes() == ["Classe_A", "Classe_B"]


def test_register_correction_appends_training_and_audit_rows(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    train_path.write_text(
        "id;sentenca;classe\n1;texto original;Classe_A\n", encoding="utf-8"
    )
    feedback_path = tmp_path / "feedback.csv"
    sentences_dir = tmp_path / "sentencas"
    sentences_dir.mkdir()
    (sentences_dir / "Classe_A.txt").write_text("a", encoding="utf-8")
    (sentences_dir / "Classe_B.txt").write_text("b", encoding="utf-8")
    monkeypatch.setattr(feedback, "TRAIN_PATH", train_path)
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(feedback, "SENTENCES_DIR", sentences_dir)

    feedback.register_correction("nova notificação", "Classe_A", "Classe_B")

    with train_path.open(encoding="utf-8", newline="") as file:
        train_rows = list(csv.DictReader(file, delimiter=";"))
    with feedback_path.open(encoding="utf-8", newline="") as file:
        audit_rows = list(csv.DictReader(file, delimiter=";"))

    assert train_rows[-1] == {
        "id": "2",
        "sentenca": "nova notificação",
        "classe": "Classe_B",
    }
    assert audit_rows[-1]["classe_predita"] == "Classe_A"
    assert audit_rows[-1]["classe_correta"] == "Classe_B"