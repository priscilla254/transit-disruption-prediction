# Data quality

Profile of `berlin_sbahn_delays.db` from `src/profile_tables.py`, before any joins. None of the seven tables declares a primary key. Uniqueness below is on the presumed key only.

## Result

Every column is 0% null. No table has an exact duplicate row. Each presumed key is unique.

| Table | Rows | Duplicate rows | Presumed key | Unique |
|---|---:|---:|---|---|
| city_events | 3 | 0 | event_name | yes |
| incidents | 36 | 0 | incident_id | yes |
| lines | 6 | 0 | line_id | yes |
| stations | 10 | 0 | station_id | yes |
| strikes | 3 | 0 | strike_start | yes |
| trips | 131,771 | 0 | trip_id | yes |
| weather | 8,761 | 0 | timestamp | yes |

`city_events` and `strikes` have no id column. Uniqueness for those tables is only on `event_name` and `strike_start`.

## Columns

Null percent is 0 for every column below. SQL type is the declared SQLite type. Pandas dtype is what `read_sql` infers. Timestamp columns are stored as text, so pandas reads them as `str`.

### city_events

| Column | SQL type | Pandas dtype | Null % |
|---|---|---|---:|
| event_name | TEXT | str | 0 |
| date | TEXT | str | 0 |
| venue_station_id | TEXT | str | 0 |
| impact_on_lines | TEXT | str | 0 |

### incidents

| Column | SQL type | Pandas dtype | Null % |
|---|---|---|---:|
| incident_id | TEXT | str | 0 |
| timestamp | TIMESTAMP | str | 0 |
| incident_type | TEXT | str | 0 |
| delay_impact_factor | REAL | float64 | 0 |
| line_id | TEXT | str | 0 |

### lines

| Column | SQL type | Pandas dtype | Null % |
|---|---|---|---:|
| line_id | TEXT | str | 0 |
| line_name | TEXT | str | 0 |
| is_ring_line | INTEGER | int64 | 0 |
| delay_propensity | REAL | float64 | 0 |

### stations

| Column | SQL type | Pandas dtype | Null % |
|---|---|---|---:|
| station_id | TEXT | str | 0 |
| name | TEXT | str | 0 |
| is_major_hub | INTEGER | int64 | 0 |
| location_category | TEXT | str | 0 |

### strikes

| Column | SQL type | Pandas dtype | Null % |
|---|---|---|---:|
| strike_start | TIMESTAMP | str | 0 |
| strike_end | TIMESTAMP | str | 0 |

### trips

| Column | SQL type | Pandas dtype | Null % |
|---|---|---|---:|
| trip_id | INTEGER | int64 | 0 |
| line_id | TEXT | str | 0 |
| start_station_id | TEXT | str | 0 |
| end_station_id | TEXT | str | 0 |
| scheduled_departure_time | TIMESTAMP | str | 0 |
| is_peak_hour | INTEGER | int64 | 0 |
| is_delayed | INTEGER | int64 | 0 |
| delay_minutes | INTEGER | int64 | 0 |
| is_cancelled | INTEGER | int64 | 0 |

### weather

| Column | SQL type | Pandas dtype | Null % |
|---|---|---|---:|
| timestamp | TIMESTAMP | str | 0 |
| temperature_c | REAL | float64 | 0 |
| precipitation_mm | REAL | float64 | 0 |
| wind_speed_kmh | REAL | float64 | 0 |
| weather_condition | TEXT | str | 0 |

## Referential integrity

`src/check_trip_references.py` checks `trips` against `lines` and `stations`. Across 131,771 trips, every foreign key hits a parent row.

| Trip column | Parent | Orphan rows |
|---|---|---:|
| line_id | lines.line_id | 0 |
| start_station_id | stations.station_id | 0 |
| end_station_id | stations.station_id | 0 |

## Date coverage

Trips run from `2024-01-01 00:01:00` through `2024-12-30 23:59:00`, on 365 distinct dates. `2024-12-31` has no trips. Every incident, city event, and strike falls inside that calendar, and each of those dates has trips.

June has no incidents. The other months have 1–5.

### incidents

| Timestamp | Type | Line |
|---|---|---|
| 2024-01-13 09:23 | Signal Failure | S1 |
| 2024-01-14 19:09 | Signal Failure | S5 |
| 2024-01-16 03:11 | Power Outage | S42 |
| 2024-01-23 15:40 | Track Maintenance | S7 |
| 2024-01-24 22:34 | Technical Fault | S41 |
| 2024-02-10 07:56 | Track Maintenance | S5 |
| 2024-02-14 20:09 | Technical Fault | S1 |
| 2024-02-21 12:58 | Track Maintenance | S2 |
| 2024-02-22 03:09 | Signal Failure | S5 |
| 2024-03-12 03:29 | Signal Failure | S5 |
| 2024-03-22 23:49 | Technical Fault | S42 |
| 2024-04-08 02:22 | Technical Fault | S7 |
| 2024-04-11 22:10 | Track Maintenance | S1 |
| 2024-04-20 11:29 | Signal Failure | S2 |
| 2024-04-21 07:56 | Power Outage | S7 |
| 2024-04-26 09:52 | Technical Fault | S41 |
| 2024-05-05 07:37 | Technical Fault | S1 |
| 2024-05-22 00:13 | Technical Fault | S7 |
| 2024-05-22 05:18 | Power Outage | S5 |
| 2024-05-22 15:28 | Technical Fault | S5 |
| 2024-07-02 11:44 | Technical Fault | S2 |
| 2024-07-04 19:42 | Power Outage | S2 |
| 2024-07-12 02:41 | Power Outage | S42 |
| 2024-07-13 03:18 | Power Outage | S1 |
| 2024-08-02 07:31 | Track Maintenance | S41 |
| 2024-08-04 01:05 | Track Maintenance | S7 |
| 2024-08-04 11:36 | Power Outage | S42 |
| 2024-08-17 20:06 | Signal Failure | S7 |
| 2024-09-15 20:32 | Power Outage | S5 |
| 2024-10-01 04:15 | Power Outage | S5 |
| 2024-10-09 10:00 | Technical Fault | S5 |
| 2024-11-05 09:01 | Technical Fault | S1 |
| 2024-11-17 21:06 | Track Maintenance | S41 |
| 2024-11-23 03:48 | Technical Fault | S42 |
| 2024-12-12 18:36 | Technical Fault | S42 |
| 2024-12-25 18:36 | Track Maintenance | S2 |

### city_events

| Date | Event | Venue station | Lines |
|---|---|---|---|
| 2024-09-06 | IFA Berlin | S8 | S41, S42, S5 |
| 2024-09-29 | Berlin Marathon | S4 | S1, S2, S41, S42 |
| 2024-10-12 | Hertha BSC Match | S5 | S5 |

### strikes

Stored as midnight timestamps. Trip counts below use a half-open window from `strike_start` up to, but not including, `strike_end`.

| Start | End | Trip days inside the window | Trips |
|---|---|---:|---:|
| 2024-03-15 00:00 | 2024-03-17 00:00 | 2 | 714 |
| 2024-06-20 00:00 | 2024-06-25 00:00 | 5 | 1,744 |
| 2024-11-05 00:00 | 2024-11-07 00:00 | 2 | 734 |


