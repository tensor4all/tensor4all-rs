- `tensor4all-core`: `floating_zone_walk` returns `FloatingZoneError<Err>`
  instead of the callback's own `Err`, and validates its arguments before
  evaluating the starting point, as `floating_zone_walk_with_initial_error`
  already did (#1320). A starting point whose length does not match
  `local_dims` used to panic on an internal index; it now returns
  `FloatingZoneError::StartingPointLength` without calling the evaluator. Call
  sites recover the callback error from `FloatingZoneError::Evaluation`.
