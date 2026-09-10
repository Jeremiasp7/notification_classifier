from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from typing import Any

from scripts.train import train_and_select

# Estados possíveis: "ocioso", "em_andamento", "concluido", "erro"
_lock = threading.Lock()

_state: dict[str, Any] = {
    "status": "ocioso",
    "mensagem": "Nenhum treinamento foi executado ainda.",
    "melhor_modelo": None,
    "resultados": None,
    "iniciado_em": None,
    "finalizado_em": None,
}


def get_status() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def is_running() -> bool:
    with _lock:
        return _state["status"] == "em_andamento"


def start_training() -> bool:
    """
    Inicia o treinamento em background, caso não haja um em andamento.
    Retorna False se já existir um treinamento em execução.
    """
    with _lock:
        if _state["status"] == "em_andamento":
            return False
        _state.update(
            {
                "status": "em_andamento",
                "mensagem": "Treinamento em andamento: gerando embeddings e "
                "treinando os modelos candidatos...",
                "melhor_modelo": None,
                "resultados": None,
                "iniciado_em": datetime.now(timezone.utc).isoformat(),
                "finalizado_em": None,
            }
        )

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return True


def _run_training() -> None:
    from app.predictor import predictor  # import local para evitar ciclo

    try:
        train_and_select(plot=True)
        predictor.reload()

        with _lock:
            metadata = predictor.metadata
            _state.update(
                {
                    "status": "concluido",
                    "mensagem": f"Treinamento concluído. Melhor modelo: "
                    f"'{metadata.get('classifier')}'.",
                    "melhor_modelo": metadata.get("classifier"),
                    "resultados": metadata.get("val_results"),
                    "finalizado_em": datetime.now(timezone.utc).isoformat(),
                }
            )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        with _lock:
            _state.update(
                {
                    "status": "erro",
                    "mensagem": f"Falha no treinamento: {exc}",
                    "finalizado_em": datetime.now(timezone.utc).isoformat(),
                }
            )
