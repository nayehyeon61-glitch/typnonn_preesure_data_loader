# Utility Metrics

실제 구현은 [`src/typhoon_pressure/utility/`](../src/typhoon_pressure/utility/)에 있습니다.

현재 제공 metric:

- dateline을 고려한 longitude wrapping
- Haversine great-circle track error (km)
- track mean error, RMSE, median, 90th percentile, maximum
- 기압·풍속 같은 scalar value의 bias, MAE, RMSE
- NaN과 명시적 mask를 제외한 valid-count 집계

```python
from typhoon_pressure.utility.metrics import track_error_statistics

metrics = track_error_statistics(
    pred_lat, pred_lon, ibtracs_lat, ibtracs_lon, mask=valid_mask
)
```
