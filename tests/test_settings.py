from rcs.settings import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings()

    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.max_image_pixels == 40_000_000
    assert settings.queue_capacity == 128
    assert settings.api_key is None
