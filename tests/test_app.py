from app.Forecasting_Fantasy_Football import get_points_data, get_playing_data, get_forecast_data


class TestApp:
    def test_get_points_data(self):
        df = get_points_data("points.pq")
        assert not df.empty

    def test_get_playing_data(self):
        df = get_playing_data("playing.pq")
        assert not df.empty

    def test_get_forecast_data(self):
        df = get_forecast_data("points.pq", "playing.pq")
        assert not df.empty
