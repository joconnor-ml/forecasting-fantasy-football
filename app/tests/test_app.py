from pathlib import Path

from app.utils import (
    read_parquet_cached,
    get_forecast_data,
)

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"


class TestApp:
    def test_get_points_data(self):
        df = read_parquet_cached(TEST_DATA_DIR / "points.pq")
        assert not df.empty

    def test_get_playing_data(self):
        df = read_parquet_cached(TEST_DATA_DIR / "playing.pq")
        assert not df.empty

    def test_get_forecast_data(self):
        df = get_forecast_data(
            TEST_DATA_DIR / "points.pq", TEST_DATA_DIR / "playing.pq", TEST_DATA_DIR / "features.pq"
        )
        assert not df.empty
