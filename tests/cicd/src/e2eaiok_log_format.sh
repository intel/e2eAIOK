#!/bin/bash

case $MODEL_NAME in
  "wnd")
    strict_log_pattern=^.*step.*binary_accuracy.*auc.*loss.*time.*$
    loose_log_pattern=^.*step.*binary_accuracy.*$
    ;;
  "dien")
    strict_log_pattern=^.*iter.*train_loss.*train_accuracy.*train_aux_loss.*train_time.*$
    loose_log_pattern=^.*iter.*train_loss.*$
    ;;
  "dlrm")
    strict_log_pattern=^.*Finished.*training.*it.*of.*epoch.*ms/it.*loss.*accuracy.*$
    loose_log_pattern=^.*Finished.*training.*$
    ;;
  *)
    # Epoch {epoch_id}, Iteration {iter_begin} to {iter_end} took {train_elapse} s, {train_per_iter} ms/iter, loss is {np.mean(losses)}, test took {test_time} secs, test_auc is {auc}"
    strict_log_pattern=^.*Epoch.*Iteration.*to.*took.*s.*ms/iter.*loss.*is.*test.*took.*secs.*test_auc.*is.*$
    loose_log_pattern=^.*Epoch.*Iteration.*$
    ;;
esac

while IFS= read -r line
do
  if [[ "$line" =~ $strict_log_pattern ]]
  then
    echo "Log pattern passed!"
  fi
done <<< $(grep $loose_log_pattern $LOG_FILE)
