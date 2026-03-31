""" 

Weather Props - Pulls Weather Data that could potentially be used 
for formulating a algo for players who have best odds for HRs? 


Setup:


"""

import urllib.request
import urllib.error
import json
from datetime import date, datetime
from dataclasses import dataclass


@dataclass
class ForecastPeriod:
    name: str           # e.g. "Tonight", "Monday", "Monday Night"
    start: datetime
    end: datetime
    is_daytime: bool
    temperature: int
    temperature_unit: str
    wind_speed: str
    wind_direction: str
    short_forecast: str
    detailed_forecast: str

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.start.strftime('%Y-%m-%d')}): "
            f"{self.temperature}°{self.temperature_unit}, "
            f"Wind {self.wind_speed} {self.wind_direction} — "
            f"{self.short_forecast}"
        )


class WeatherGovClient:
    """
    Fetches weather forecasts from the National Weather Service API (weather.gov).

    Usage:
        client = WeatherGovClient(latitude=38.8894, longitude=-77.0352)
        periods = client.get_forecast_for_date(date(2026, 4, 1))
        for p in periods:
            print(p)
    """

    BASE_URL = "https://api.weather.gov"

    def __init__(self, latitude: float, longitude: float, user_agent: str = "weather-client/1.0 (contact@example.com)"):
        self.latitude = latitude
        self.longitude = longitude
        self.user_agent = user_agent
        self._grid_info: dict | None = None

    def _request(self, url: str) -> dict:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/geo+json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} fetching {url}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error fetching {url}: {e.reason}") from e

    def _get_grid_info(self) -> dict:
        """Resolve lat/lon to NWS grid info (cached after first call)."""
        if self._grid_info is None:
            url = f"{self.BASE_URL}/points/{self.latitude:.4f},{self.longitude:.4f}"
            data = self._request(url)
            self._grid_info = data["properties"]
        return self._grid_info

    def get_forecast(self, hourly: bool = False) -> list[ForecastPeriod]:
        """
        Fetch the full forecast (up to ~7 days).

        Args:
            hourly: If True, return hourly periods instead of 12-hour periods.

        Returns:
            List of ForecastPeriod objects sorted by start time.
        """
        grid = self._get_grid_info()
        url = grid["forecastHourly"] if hourly else grid["forecast"]
        data = self._request(url)
        return [self._parse_period(p) for p in data["properties"]["periods"]]

    def get_forecast_for_date(self, target_date: date, hourly: bool = False) -> list[ForecastPeriod]:
        """
        Get all forecast periods that fall on the given date.

        Args:
            target_date: The date to filter forecasts for.
            hourly:      If True, use the hourly forecast endpoint.

        Returns:
            List of ForecastPeriod objects whose start time matches target_date.
            Returns an empty list if the date is outside the forecast window (~7 days).
        """
        periods = self.get_forecast(hourly=hourly)
        return [p for p in periods if p.start.date() == target_date]

    @staticmethod
    def _parse_period(period: dict) -> ForecastPeriod:
        return ForecastPeriod(
            name=period["name"],
            start=datetime.fromisoformat(period["startTime"]),
            end=datetime.fromisoformat(period["endTime"]),
            is_daytime=period["isDaytime"],
            temperature=period["temperature"],
            temperature_unit=period["temperatureUnit"],
            wind_speed=period.get("windSpeed", ""),
            wind_direction=period.get("windDirection", ""),
            short_forecast=period.get("shortForecast", ""),
            detailed_forecast=period.get("detailedForecast", ""),
        )

    @property
    def location_info(self) -> dict:
        """Returns city, state, timezone, and grid metadata for this location."""
        grid = self._get_grid_info()
        return {
            "city": grid.get("relativeLocation", {}).get("properties", {}).get("city"),
            "state": grid.get("relativeLocation", {}).get("properties", {}).get("state"),
            "timezone": grid.get("timeZone"),
            "grid_id": grid.get("gridId"),
            "grid_x": grid.get("gridX"),
            "grid_y": grid.get("gridY"),
        }


if __name__ == "__main__":
    # Example: forecast for Washington D.C. on a specific date
    client = WeatherGovClient(latitude=38.8894, longitude=-77.0352)

    info = client.location_info
    print(f"Location: {info['city']}, {info['state']} (timezone: {info['timezone']})\n")

    target = date(2026, 3, 31)
    periods = client.get_forecast_for_date(target)

    if not periods:
        print(f"No forecast available for {target} (outside the ~7-day window).")
    else:
        print(f"Forecast for {target}:")
        for p in periods:
            print(f"  {p}")
            if p.detailed_forecast:
                print(f"    {p.detailed_forecast}")
