import logging

from core.runtime import configure_logging


def test_huggingface_unauthenticated_warning_is_filtered(tmp_path) -> None:
    configure_logging(tmp_path)

    hf_logger = logging.getLogger("huggingface_hub.utils._http")
    hf_logger.warning(
        "Warning: You are sending unauthenticated requests to the HF Hub. "
        "Please set a HF_TOKEN to enable higher rate limits and faster downloads."
    )
    hf_logger.warning("A different Hugging Face warning")

    logging.shutdown()

    log_text = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "unauthenticated requests to the HF Hub" not in log_text
    assert "A different Hugging Face warning" in log_text
