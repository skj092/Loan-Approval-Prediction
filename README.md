# Loan-Approval-Prediction


Dataset Description

Files
train.csv (id, person_age, person_income, person_home_ownership, person_emp_len, loan_intent, loan_grade, load_amnt, loan_int_rate, load_percent, income, cb_person_cred_hist_length, loan_status)
- the training dataset; loan_status is the binary target
test.csv - the test dataset; your objective is to predict probability of the target loan_status for each row
sample_submission.csv - a sample submission file in the correct format

Evaluation:
Submissions are evaluated using area under the ROC curve using the predicted probabilities and the ground truth targets.

Submission File:
For each id row in the test set, you must predict target loan_status. The file should contain a header and have the following format:

```csv
id,loan_status
58645,0.5
58646,0.5
58647,0.5
etc.
```


