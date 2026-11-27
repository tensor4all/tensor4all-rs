- `tensor4all-treeaci`: `TreeAciError::RollbackFailed { error, rollback }` (#1464).
  When a step fails and undoing its partial state fails too, the step's own error
  is returned in `error` with the rollback failure in `rollback`; before, the
  rollback error replaced the step's error.
