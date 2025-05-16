import numpy as np

# Sequence numbers
seq = [1, 2, 8, 11, 12, 13, 14, 15, 21, 23]

# Season labels
season = ["Fall", "fall", "fall", "fall", "fall", "spring", "spring", "spring", "spring", "spring"]

# True scale values
true_scale = [17.515, 9.989, 13.310, 12.203, 39.530, 6.696, 23.535, 5.887, 16.906, 4.790]

# Corrected predicted scale values (these were after applying lambda = 1.5)
corrected_pred = [19.265, 9.333, 12.325, 13.011, 33.110, 5.508, 13.256, 1.828, 11.947, 2.871]

# Percentage difference (corrected)
percent_diff_corr = [9.99, 6.57, 7.40, 6.62, 16.24, 17.74, 43.67, 68.94, 29.33, 40.07]

orig_val=[]
for val in corrected_pred:
    orig_val.append(val/1.5)

corr_val = []
for true_val, pred_val in zip(true_scale, orig_val):
    corr_val.append(true_val/pred_val)

print(corr_val)
median_corr = np.median(corr_val)
print(median_corr)

new_pred=[]
for pred_val in orig_val:
    new_pred.append(pred_val*median_corr)

orig_diff=[]
for p_val, t_val in zip(orig_val, true_scale):
    orig_diff.append(((p_val-t_val)/t_val)*100)

adjusted_p = []
for p in orig_val:
    adjusted_p.append(median_corr*p)

adjusted_diff = []
for p_val, t_val in zip(adjusted_p, true_scale):
    adjusted_diff.append(((p_val-t_val)/t_val)*100)


for i in range(len(true_scale)):
    env = season[i]
    curr_seq = seq[i]
    t_scale = true_scale[i]
    p_scale = orig_val[i]
    o_diff = orig_diff[i]
    a_p = adjusted_p[i]
    a_d = adjusted_diff[i]

    # percent_error = abs((predicted_scale - scale_factor) / scale_factor * 100)

    print(f"{curr_seq} & {env} & {t_scale:.3f} & {p_scale:.2f} & {o_diff:.2f} & {a_p:.2f} & {a_d:.2f} \\\\")





