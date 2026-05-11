"""Price-based features: returns, momentum, vol, regime, GARCH."""

import numpy as np
from arch import arch_model

from miner.features.base import FeatureEngine


class PriceFeatures(FeatureEngine):
    """Computes technical features from a price series.

    Features (per asset):
    - Log returns at multiple horizons (1, 5, 15, 60, 240 bars)
    - Rolling mean/std/skew of returns (windows: 60, 240, 720)
    - GARCH(1,1) conditional volatility
    - RSI (14-bar)
    - Bollinger band position
    - Volume z-score
    - Price momentum percentile
    - Hour-of-day cyclic encoding
    """

    warmup = 720  # need at least 720 bars for rolling windows

    # Fixed output size: 25 (price) + 3 (FRED macro) + 1 (GNews crypto) = 29
    OUTPUT_DIM = 29

    def compute(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
        fred: dict[str, float] | None = None,
        gnews_crypto: float = 0.0,
    ) -> np.ndarray:
        """Compute all price features.

        Args:
            prices: 1-D close price array (most recent last)
            volumes: optional 1-D volume array
            fred: optional dict of FRED macro values
            gnews_crypto: GNews crypto sentiment score in [-1, 1]

        Returns:
            1-D feature vector (29 dims)
        """
        prices = np.asarray(prices, dtype=float)
        eps = 1e-12

        if len(prices) < self.warmup:
            return np.zeros(self.OUTPUT_DIM, dtype=float)

        features = []

        # --- Log returns at multiple horizons ---
        log_ret = np.diff(np.log(prices + eps))
        for h in [1, 5, 15, 60, 240]:
            ret_h = np.sum(log_ret[-h:]) if len(log_ret) >= h else 0.0
            features.append(ret_h)

        # --- Rolling stats of returns ---
        for w in [60, 240, 720]:
            window = log_ret[-w:] if len(log_ret) >= w else log_ret
            features.append(np.mean(window))
            features.append(np.std(window))
            if len(window) > 2:
                m = np.mean(window)
                features.append(np.mean((window - m) ** 3) / (np.std(window) ** 3 + eps))
            else:
                features.append(0.0)

        # --- GARCH(1,1) conditional volatility ---
        try:
            if len(log_ret) > 200:
                am = arch_model(log_ret[-500:], vol="Garch", p=1, q=1, dist="normal", mean="Zero")
                res = am.fit(disp="off", show_warning=False)
                cond_vol = np.sqrt(res.conditional_volatility[-1])
                features.append(cond_vol)
            else:
                features.append(np.std(log_ret))
        except Exception:
            features.append(np.std(log_ret[-60:]))

        # --- RSI (14-bar) ---
        delta = np.diff(prices[-15:])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains) + eps
        avg_loss = np.mean(losses) + eps
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100 - 0.5)  # center around 0

        # --- Bollinger band position ---
        ma20 = np.mean(prices[-20:])
        std20 = np.std(prices[-20:]) + eps
        bb_pos = (prices[-1] - ma20) / (2 * std20)
        features.append(bb_pos)

        # --- MACD ---
        if len(prices) > 35:
            ema12 = _ema(prices[-35:], 12)
            ema26 = _ema(prices[-35:], 26)
            macd = ema12 - ema26
            features.append(macd / (prices[-1] + eps))
        else:
            features.append(0.0)

        # --- Volume z-score ---
        if volumes is not None and len(volumes) > 60:
            vol_recent = volumes[-60:]
            vol_z = (volumes[-1] - np.mean(vol_recent)) / (np.std(vol_recent) + eps)
            features.append(vol_z)
        else:
            features.append(0.0)

        # --- Momentum percentile ---
        if len(log_ret) > 60:
            current_ret = log_ret[-1]
            features.append(np.mean(log_ret[-60:] < current_ret))
        else:
            features.append(0.5)

        # --- Hour-of-day (sin/cos) ---
        # This is a placeholder — the caller should pass actual timestamps
        features.append(0.0)
        features.append(0.0)

        # --- Regime features ---
        # Rolling Sharpe (annualized)
        if len(log_ret) > 240:
            ret_240 = log_ret[-240:]
            sharpe = np.mean(ret_240) / (np.std(ret_240) + eps) * np.sqrt(525600)
            features.append(sharpe)
        else:
            features.append(0.0)

        # Vol-of-vol
        if len(log_ret) > 120:
            rolling_vol = np.array([np.std(log_ret[i - 60 : i]) for i in range(60, len(log_ret))])
            if len(rolling_vol) > 1:
                features.append(np.std(rolling_vol) / (np.mean(rolling_vol) + eps))
            else:
                features.append(0.0)
        else:
            features.append(0.0)

        # Return autocorrelation (lag-1)
        if len(log_ret) > 60:
            r = log_ret[-60:]
            features.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        else:
            features.append(0.0)

        # --- FRED macro context (3 dims) ---
        fred = fred or {}
        features.append(fred.get("fed_funds_rate", 0.0))
        features.append(fred.get("treasury_spread_10y2y", 0.0))
        features.append(fred.get("dxy_broad", 0.0))

        # --- GNews crypto sentiment (1 dim) ---
        features.append(gnews_crypto)

        return np.array(features, dtype=float)


def _ema(data: np.ndarray, span: int) -> float:
    """Compute exponential moving average."""
    alpha = 2.0 / (span + 1)
    ema = data[0]
    for val in data[1:]:
        ema = alpha * val + (1 - alpha) * ema
    return ema
