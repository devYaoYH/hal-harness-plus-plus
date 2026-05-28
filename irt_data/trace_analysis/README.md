# Trace analysis: taubench_airline task 20

Runs analyzed: 47 (23 success, 24 failure).

## Task

Your user id is james_taylor_7043. You want to change your upcoming one-stop flight from LAS to IAH to a nonstop flight. Your reservation ID is 1N99U6. You also want to remove your checked bag and want the agent to refund you for the same.

Expected action signatures:

get_reservation_details(reservation_id=1N99U6) -> search_direct_flight(origin=LAS) -> update_reservation_flights(HAT266+HAT112)

## Strongest suffix separators

| suffix | support | success rate | log-odds lift |
| --- | ---: | ---: | ---: |
| `respond -> respond -> respond -> respond` | 6 | 0.00 | -2.80 |
| `respond -> respond -> respond -> respond -> respond` | 4 | 0.00 | -2.33 |
| `respond -> respond -> respond -> respond -> respond -> respond -> respond` | 3 | 0.00 | -2.03 |
| `respond -> respond -> respond -> respond -> respond -> respond` | 3 | 0.00 | -2.03 |
| `update_reservation_flights -> respond -> respond -> respond` | 8 | 0.88 | 1.96 |
| `respond -> update_reservation_flights -> respond -> respond -> respond` | 7 | 0.86 | 1.76 |
| `respond -> search_direct_flight -> respond -> respond -> respond -> respond -> update_reservation_flights -> respond` | 2 | 1.00 | 1.74 |
| `search_direct_flight -> respond -> respond -> respond -> respond -> update_reservation_flights -> respond` | 2 | 1.00 | 1.74 |
| `search_direct_flight -> respond -> update_reservation_flights -> respond -> respond -> respond` | 2 | 1.00 | 1.74 |
| `respond -> respond -> respond -> respond -> update_reservation_flights -> respond` | 2 | 1.00 | 1.74 |
| `search_direct_flight -> respond -> respond -> update_reservation_flights -> respond -> update_reservation_flights -> respond` | 2 | 0.00 | -1.65 |
| `respond -> respond -> update_reservation_flights -> respond -> update_reservation_flights -> respond` | 2 | 0.00 | -1.65 |

## Strongest signature suffix separators

| suffix | support | success rate | log-odds lift |
| --- | ---: | ---: | ---: |
| `update_reservation_flights(HAT266+HAT112) -> respond -> respond -> respond` | 7 | 1.00 | 3.10 |
| `respond -> update_reservation_flights(HAT266+HAT112) -> respond -> respond -> respond` | 6 | 1.00 | 2.90 |
| `respond -> respond -> respond -> respond` | 6 | 0.00 | -2.80 |
| `respond -> update_reservation_flights(HAT266+HAT112) -> respond -> respond` | 5 | 1.00 | 2.68 |
| `update_reservation_flights(HAT266+HAT112) -> respond -> respond` | 5 | 1.00 | 2.68 |
| `respond -> respond -> respond -> respond -> respond` | 4 | 0.00 | -2.33 |
| `respond -> respond -> respond -> respond -> respond -> respond -> respond` | 3 | 0.00 | -2.03 |
| `respond -> respond -> respond -> respond -> respond -> respond` | 3 | 0.00 | -2.03 |
| `respond -> search_direct_flight(origin=LAS) -> respond -> respond -> respond -> respond -> update_reservation_flights(HAT266+HAT112) -> respond` | 2 | 1.00 | 1.74 |
| `search_direct_flight(origin=LAS) -> respond -> respond -> respond -> respond -> update_reservation_flights(HAT266+HAT112) -> respond` | 2 | 1.00 | 1.74 |
| `search_direct_flight(origin=LAS) -> respond -> update_reservation_flights(HAT266+HAT112) -> respond -> respond -> respond` | 2 | 1.00 | 1.74 |
| `respond -> respond -> respond -> respond -> update_reservation_flights(HAT266+HAT112) -> respond` | 2 | 1.00 | 1.74 |

## Feature cuts

| feature | value | support | success rate | successes | failures |
| --- | --- | ---: | ---: | ---: | ---: |
| `contains_expected_signature_subsequence` | `0` | 11 | 0.00 | 0 | 11 |
| `contains_expected_action_subsequence` | `0` | 4 | 0.00 | 0 | 4 |
| `n_search_direct_flight` | `0` | 4 | 0.00 | 0 | 4 |
| `n_update_reservation_flights` | `0` | 4 | 0.00 | 0 | 4 |
| `used_expected_update_flight` | `0` | 4 | 0.00 | 0 | 4 |
| `has_update_reservation_baggages` | `1` | 3 | 0.00 | 0 | 3 |
| `all_updates_are_expected` | `0` | 14 | 0.14 | 2 | 12 |
| `first_update_is_expected` | `0` | 14 | 0.14 | 2 | 12 |
| `n_search_direct_flight` | `2+` | 7 | 0.14 | 1 | 6 |
| `used_non_expected_update_flight` | `1` | 10 | 0.20 | 2 | 8 |
| `contains_expected_signature_subsequence` | `1` | 36 | 0.64 | 23 | 13 |
| `all_updates_are_expected` | `1` | 33 | 0.64 | 21 | 12 |
| `first_update_is_expected` | `1` | 33 | 0.64 | 21 | 12 |
| `n_update_reservation_flights` | `2+` | 8 | 0.38 | 3 | 5 |
| `n_search_direct_flight` | `1` | 36 | 0.61 | 22 | 14 |
| `n_update_reservation_flights` | `1` | 35 | 0.57 | 20 | 15 |
| `used_non_expected_update_flight` | `0` | 37 | 0.57 | 21 | 16 |
| `contains_expected_action_subsequence` | `1` | 43 | 0.53 | 23 | 20 |

## Files

- `task_trajectory_summary.csv`: one row per trace/run.
- `suffix_patterns.csv`: suffix trie counts over action names.
- `signature_suffix_patterns.csv`: suffix trie counts over action signatures.
- `feature_cuts.csv`: coarse success/failure cuts for interpretable factors.
- `selected_task.json`: target task metadata.
