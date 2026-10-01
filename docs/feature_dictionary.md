# Feature dictionary

For each column, ask: standing on the platform just before this trip's `scheduled_departure_time`, would this value already be known?

If the value depends on this trip's own outcome, it is leakage. Leakage stays off the feature list. The target columns are listed here so they are not reused as inputs.

## Not features

| Column | Why it is excluded |
|---|---|
| `trips.is_delayed` | Target. It is 1 only when `delay_minutes` is 6 or more, so it is this trip's outcome. |
| `trips.delay_minutes` | The same outcome in minutes. Using it predicts the label from the label. |
| `trips.is_cancelled` | Also an outcome. Cancelled trips are dropped from this use case, so the column is a row filter, not a feature and not the target. |
| `trips.trip_id` | Row id only. |
| `incidents.delay_impact_factor` | Numeric weight on the incident. It is not known on the platform, and it can hand the model the delay answer during training. Do not join it onto trips. |
| `lines.delay_propensity` | Line score from 0.6 to 1.0 that ranks how delay-prone the line is. It tracks the delay label, so a model can put it at the top of a SHAP ranking and look accurate while using `y` in disguise. Do not join it onto trips. |

## Known before departure

These come from the timetable or from a lookup that does not use this trip's result.

| Feature | Source | Why it is already known |
|---|---|---|
| `line_id` | `trips` | Printed on the departure board. |
| `start_station_id` | `trips` | The platform you are standing on. |
| `end_station_id` | `trips` | The scheduled destination. |
| `scheduled_departure_time` | `trips` | The scheduled time on the board. |
| `is_peak_hour` | `trips` | Follows from the scheduled time. |
| `line_name` | `lines` via `line_id` | The line name on the board. |
| `is_ring_line` | `lines` via `line_id` | Fixed property of the line. S41 and S42 are the ring. |
| `start_name`, `end_name` | `stations` via start and end ids | Station name. Prefixed because the station table is joined twice. |
| `start_is_major_hub`, `end_is_major_hub` | `stations` via start and end ids | Fixed property of the station. |
| `start_location_category`, `end_location_category` | `stations` via start and end ids | Fixed property of the station. |

## Known only if the timing matches

| Feature | Rule |
|---|---|
| `temperature_c`, `precipitation_mm`, `wind_speed_kmh`, `weather_condition` | Floor `scheduled_departure_time` to the hour and use that `weather.timestamp`. A later hour is not yet known. |
| `has_incident`, `incident_type` | Use the incident only when `incidents.line_id` is this trip's line and `incidents.timestamp` falls in the 60 minutes up to and including departure: `timestamp <= departure < timestamp + 60 minutes`. Leave `delay_impact_factor` off the row. |
| `is_event_day`, `event_name` | Use the event only when `city_events.date` is the departure date and `impact_on_lines` includes this line. The event is scheduled ahead of the trip. |
| `is_strike_day` | Use it only when `scheduled_departure_time` falls in `[strike_start, strike_end)`. The stored windows are midnight ranges: 2024-03-15 to 2024-03-17, 2024-06-20 to 2024-06-25, and 2024-11-05 to 2024-11-07. |

## Derived before departure

These are calculated from the joined trip table. `network_load` counts scheduled trips. The lag rates use only trips whose departure is strictly earlier, so a trip does not see its own delay or a trip leaving at the same moment.

| Feature | Rule |
|---|---|
| `hour`, `day_of_week`, `month`, `is_weekend` | Taken from `scheduled_departure_time`. Monday is 0. Saturday and Sunday are the weekend. |
| `route_key` | `line_id`, `start_station_id`, and `end_station_id` joined with a vertical bar. |
| `location_pair` | `start_location_category`, then ` to `, then `end_location_category`. |
| `hub_pattern` | `both`, `start`, `end`, or `neither`, from the stored `start_is_major_hub` and `end_is_major_hub` flags. |
| `is_raining`, `rain_at_peak` | `precipitation_mm > 0`. `rain_at_peak` also requires `is_peak_hour`. |
| `network_load` | How many trips in this table are scheduled in the same clock hour, including this trip. Delay does not change the count. |
| `line_trips_prev_hour`, `line_delay_rate_prev_hour`, `line_mean_delay_prev_hour` | Same line, departures in the hour before this trip: `[t - 1 hour, t)`. |
| `line_trips_prev_day`, `line_delay_rate_prev_day`, `line_mean_delay_prev_day` | Same line, departures in the 24 hours before this trip: `[t - 24 hours, t)`. |
| `start_station_trips_prev_hour`, `start_station_delay_rate_prev_hour` | Same start station, departures in `[t - 1 hour, t)`. |

When a lag window contains no earlier trips, the count is 0 and the rate and mean are null.
