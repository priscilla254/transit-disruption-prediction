# Model card

No model has been trained yet.

## Intended use

Estimate whether a Berlin S-Bahn trip will be delayed, using only values that would already be known on the platform before `scheduled_departure_time`. The records are synthetic. They are not a description of real S-Bahn Berlin service.

## Target

`is_delayed` is 1 exactly when `delay_minutes` is 6 or more. Cancelled trips are dropped before modeling, so `is_cancelled` is a row filter rather than a second target.

## Training data

Not assembled yet. The raw Kaggle extract is described in `docs/data_quality.md`. Columns that must not be used as inputs are listed in `docs/feature_dictionary.md`.

## Metrics

Not chosen in code yet. Accuracy is a poor summary: after cancellations are dropped, on-time trips outnumber delayed trips by 32.2 to 1.

## Limits

Disruption context is sparse: 36 incidents, 3 city events, and 3 strikes. An event-day feature would rest on three dates. `delay_propensity` and `delay_impact_factor` are excluded because they can stand in for the delay label.
