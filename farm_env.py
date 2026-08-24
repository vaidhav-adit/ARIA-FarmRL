# Seasonal farm simulator for irrigation reinforcement learning: phased weather, soil, tank, growth, rewards.


from __future__ import annotations  # Allow forward references in type hints across the file.

import numpy as np  # Numerical arrays, random Generator, clipping, exponentials.

from growth_curve import expected_growth  # Import the interpolated target growth curve by day.


def _normalize_weather(w: dict) -> np.ndarray:  # Convert raw weather scalars into four features in [0, 1].
    return np.array(  # Build a length-4 float32 feature vector used inside the state.
        [
            min(w["temperature"], 50) / 50.0,  # Cap hot days at 50 C then scale to [0, 1] temperature channel.
            w["humidity"] / 100.0,  # Relative humidity as a percent → divide by 100.
            min(w["rain_now"], 50) / 50.0,  # Cap intense instantaneous rain before scaling.
            min(w["rain_24h"], 50) / 50.0,  # Same cap for rolling 24 h rain forecast channel.
        ],
        dtype=np.float32,  # Match neural-network-friendly dtype expected by agents.
    )


class FarmEnv:  # One episode = 90 simulated days with discrete irrigation decisions.
    N_ACTIONS = 2  # Binary action: 0 do nothing, 1 run pump.
    N_STATE = 8  # Eight-dimensional observation vector (see _get_state).
    EPISODE_DAYS = 90  # Fixed horizon matching one growing season.

    def __init__(self, seed: int | None = None):  # Construct environment; seed makes weather reproducible.
        self.rng = np.random.default_rng(seed)  # NumPy Generator instance for phase noise and rain draws.
        self.day = 0  # Current day index inside the episode (starts at 0 after reset).
        self.soil = 0.5  # Initial volumetric moisture proxy in [0, 1] (50 % field proxy).
        self.water_level = 1.0  # Tank fraction full; 1.0 means 100 % before any irrigation draws.
        self.growth = expected_growth(0)  # Start biomass score aligned with calendar day 0 expectation.
        self.done = False  # Terminal flag; True once 90 transitions have completed.
        self.weather: dict = {}  # Latest human-readable weather dict (temperature, humidity, rain, …).
        self.weather_norm = np.zeros(4, dtype=np.float32)  # Cached normalized weather vector for the current day.
        self._update_weather_for_day()  # Sample weather immediately so the first state is consistent.

    def _phase_bases(self) -> tuple[float, float]:  # Return mean temperature (C) and humidity (%) for the active phase.
        d = self.day  # Read current day before it advances inside step().
        if d < 30:  # First third of season — mild profile.
            return 25.0, 50.0  # Baseline temp and humidity before Gaussian jitter.
        if d < 60:  # Middle third — hot dry stretch.
            return 35.0, 20.0  # Harsher baseline encouraging irrigation.
        return 22.0, 80.0  # Final third — cooler, wetter profile.

    def _sample_rain(self) -> tuple[float, float]:  # Draw non-negative rain_now and rain_24h with phase-dependent means.
        d = self.day  # Phase depends on the same day counter as temperature bases.
        if d < 30:  # Light precipitation season segment.
            rain_now = max(0.0, float(self.rng.normal(1.5, 2.5)))  # Random mm-class draw clipped positive.
            rain_24h = max(0.0, float(self.rng.normal(6.0, 6.0)))  # Heavier rolling total variance.
        elif d < 60:  # Dry window — rarely large rain spikes.
            rain_now = max(0.0, float(self.rng.normal(0.2, 1.2)))  # Small mean keeps most days clear.
            rain_24h = max(0.0, float(self.rng.normal(1.0, 3.0)))  # Low cumulative expectation.
        else:  # Wet window — heavier tails for storms.
            rain_now = max(0.0, float(self.rng.normal(10.0, 6.0)))  # Frequent storms possible.
            rain_24h = max(0.0, float(self.rng.normal(28.0, 12.0)))  # Large multi-hour totals.
        return rain_now, rain_24h  # Tuple fed into the weather dict.

    def _update_weather_for_day(self) -> None:  # Regenerate both raw and normalized weather for the current day index.
        base_t, base_h = self._phase_bases()  # Pull deterministic seasonal anchors.
        temp = float(base_t + self.rng.normal(0.0, 2.0))  # Jitter temperature around the phase mean.
        humidity = float(np.clip(base_h + self.rng.normal(0.0, 5.0), 5.0, 99.0))  # Keep relative humidity realistic.
        rain_now, rain_24h = self._sample_rain()  # Stochastic rainfall pair for this day.
        self.weather = {  # Pack scalars into a dict for debugging or logging if needed.
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
            "rain_now": round(rain_now, 2),
            "rain_24h": round(rain_24h, 2),
            "status": "phased_sim",
        }
        self.weather_norm = _normalize_weather(self.weather)

    def reset(self) -> np.ndarray:  # Start a fresh season; returns the initial observation vector.
        self.day = 0  # Rewind calendar.
        self.soil = 0.5  # Mid moisture baseline again.
        self.water_level = 1.0  # Refill tank for fairness across episodes.
        self.growth = expected_growth(0)  # Synchronize modeled crop with day zero target.
        self.done = False  # Episode open for interaction.
        self._update_weather_for_day()  # Draw first-day weather after reset.
        return self._get_state()  # Return eight floats describing initial situation.

    def _get_state(self) -> np.ndarray:  # Assemble observation consumed by all agents.
        return np.array(  # Fixed ordering must stay consistent with agent networks.
            [
                self.soil,  # Normalized soil moisture (first state component).
                self.water_level,  # Tank fullness fraction after any prior irrigation.
                self.growth,  # Crop development score in [0, 1].
                self.day / 90.0,  # Fraction of season elapsed (maps 0 … 90 days to 0 … 1).
                float(self.weather_norm[0]),  # Normalized temperature feature.
                float(self.weather_norm[1]),  # Normalized humidity feature.
                float(self.weather_norm[2]),  # Normalized current rain contribution.
                float(self.weather_norm[3]),  # Normalized 24 h rain outlook.
            ],
            dtype=np.float32,  # Single-precision vector for PyTorch agents.
        )

    def _growth_factors(self, rain_norm: float) -> tuple[float, float, float]:  # Decompose daily growth multiplier.
        t_norm = self.day / 90.0  # Progress through season as fraction for the phenology bell curve.
        D = float(np.exp(-((t_norm - 0.5) ** 2) / 0.157))  # Phenology bell curve peaking mid-season.
        R = 0.85 if rain_norm > 0.8 else 1.0  # Storm damage factor when normalized rain is very high.

        T_n = float(self.weather_norm[0])  # Re-read temperature feature for stress blend.
        H_n = float(self.weather_norm[1])  # Re-read humidity feature for stress blend.
        H_soil = float(np.clip(1.0 - abs(self.soil - 0.55) / 0.55, 0.0, 1.0))  # Moisture comfort around 0.55 optimum.
        T_fav = float(np.clip(1.0 - abs(T_n - 0.5) * 1.4, 0.0, 1.0))  # Mid-range normalized temp is most favorable.
        V_hum = H_n  # Humidity enters the blend directly as humidity_norm.
        inner = (  # Weighted generalised mean of favorable fractional signals.
            0.5 * np.sqrt(H_soil) + 0.3 * np.sqrt(T_fav) + 0.2 * np.sqrt(max(V_hum, 0.0))
        )
        S = float(np.clip(inner**2, 0.0, 1.0))
        return D, R, S

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:  # Apply one day of dynamics once an action is chosen.
        assert action in (0, 1), f"Invalid action: {action}"
        assert not self.done, "Episode finished; call reset()."

        T_n = float(self.weather_norm[0])  # Normalized temperature for evaporation accounting.
        H_n = float(self.weather_norm[1])  # Normalized humidity (dry air increases evaporation term).
        rain_n = float(self.weather_norm[2])  # Normalized immediate rain intensity for infiltration and storm gate.

        evaporation = 0.05 + 0.08 * T_n + 0.04 * (1.0 - H_n)  # Daily moisture loss before irrigation/rain.
        rain_gain = rain_n * 0.3  # Converts rain feature into soil moisture bump capped by scaling constant.

        pump_ok = action == 1 and self.water_level > 0.05  # Successful irrigation requires leftover tank volume.
        empty_attempt = action == 1 and self.water_level <= 0.05  # Trying to pump an almost empty tank.

        if pump_ok:  # Legitimate irrigation adds water and spends tank inventory.
            irrigation = 0.20  # Moisture increment from running the pump this day.
            self.water_level = max(0.0, self.water_level - 0.05)  # Five percent tank draw per successful actuation.
        else:  # No irrigation benefit when idle or tank too low.
            irrigation = 0.0  # Zero hydraulic addition.

        self.soil = float(
            np.clip(self.soil - evaporation + rain_gain + irrigation, 0.0, 1.0)
        )  # Physics update then clamp to physical moisture bounds.

        D, R, S = self._growth_factors(rain_n)  # Bell curve, storm gate, and combined stress multiplier.
        dg = 0.016 * D * R * S  # Max incremental growth rate modulated by all factors.
        self.growth = float(np.clip(self.growth + dg, 0.0, 1.0))  # Accumulate growth but never exceed 1.

        soil_for_penalty = self.soil  # Snapshot moisture after update for drought idle penalty logic.

        self.day += 1  # Advance calendar to next day index.
        done = self.day >= self.EPISODE_DAYS  # Episode ends after 90 transitions.

        if not done:  # Roll the weather dice for the upcoming day unless episode just finished.
            self._update_weather_for_day()  # Refresh both raw dict and normalized vector.

        next_state = self._get_state()  # Observation exposes post-step soil, tank, growth, calendar, weather.
        self.done = done  # Cache termination for external loops.

        expected = expected_growth(self.day)  # Target biomass for the new calendar day after increment.

        reward = 0.0  # Accumulate scalar feedback defined by the curriculum below.
        if self.growth < expected:  # Under-performing vs agronomic schedule.
            reward -= 5.0 * float(expected - self.growth)  # Penalty proportional to shortfall magnitude.
        elif self.growth > expected:  # Out-performing schedule earns mild bonus to avoid discouraging healthy plants.
            reward += 1.0 * float(self.growth - expected)  # Small positive signal for surplus growth.

        if pump_ok:  # Operating the pump carries minor cost to discourage trivial spraying.
            reward -= 0.05  # Fixed operational penalty per valid irrigation.
        if empty_attempt:  # Hitting pump with insufficient water is heavily discouraged.
            reward -= 5.0  # Large penalty discouraging hardware misuse.

        if soil_for_penalty < 0.30 and action == 0:  # Letting soil dry critically while idling is bad.
            reward -= 2.0  # Drought negligence penalty.

        if done and self.growth >= 0.80:  # Successful commercial-grade finish yields jackpot reward.
            reward += 100.0  # Harvest bonus when season ends strong enough.

        reward = float(round(reward, 4))  # Stabilize floating noise for logging and value learning.

        return next_state, reward, done  # Standard Gym-style transition tuple.
