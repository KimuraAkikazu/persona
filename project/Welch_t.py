import numpy as np
from scipy import stats

# 各チームの正解率データ (ご提示のJSONから、または analysis_gemini.py の出力ファイルから取得)
accuracies_team_mixed = [
    0.78, 0.68, 0.68, 0.72, 0.68, 0.68, 0.66, 0.76, 0.68, 0.7,
    0.68, 0.76, 0.74, 0.74, 0.7, 0.64, 0.68, 0.74, 0.72, 0.66,
    0.66, 0.72, 0.66, 0.74, 0.7, 0.64, 0.62, 0.72, 0.6, 0.76,
    0.66, 0.64, 0.72, 0.68, 0.72, 0.72, 0.74, 0.68, 0.7, 0.7,
    0.7, 0.74, 0.64, 0.74, 0.66, 0.74, 0.78, 0.64, 0.7, 0.74
]

accuracies_team_none = [
    0.66, 0.6, 0.8, 0.64, 0.74, 0.64, 0.72, 0.68, 0.7, 0.72,
    0.74, 0.64, 0.64, 0.7, 0.68, 0.74, 0.62, 0.68, 0.7, 0.6,
    0.66, 0.72, 0.64, 0.68, 0.68, 0.64, 0.66, 0.74, 0.76, 0.74,
    0.66, 0.68, 0.76, 0.68, 0.72, 0.72, 0.62, 0.7, 0.68, 0.62,
    0.64, 0.7, 0.76, 0.7, 0.66, 0.68, 0.7, 0.72, 0.76, 0.68
]

# 平均値と標準偏差の確認 (任意)
mean_mixed = np.mean(accuracies_team_mixed)
std_mixed = np.std(accuracies_team_mixed, ddof=1)
mean_none = np.mean(accuracies_team_none)
std_none = np.std(accuracies_team_none, ddof=1)

print(f"TeamMixed: Mean={mean_mixed:.4f}, StdDev={std_mixed:.4f}, N={len(accuracies_team_mixed)}")
print(f"TeamNone:  Mean={mean_none:.4f}, StdDev={std_none:.4f}, N={len(accuracies_team_none)}")

# 2サンプルのt検定 (Welchのt検定を推奨: equal_var=False)
t_statistic, p_value = stats.ttest_ind(
    accuracies_team_mixed,
    accuracies_team_none,
    equal_var=False  # Welchのt検定
)

print(f"\nIndependent Two-Sample t-test (Welch's t-test):")
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# 結果の解釈
alpha = 0.05  # 有意水準
if p_value < alpha:
    print(f"p-value ({p_value:.4f}) < alpha ({alpha}), "
          "therefore we reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference "
          "in mean accuracies between TeamMixed and TeamNone.")
else:
    print(f"p-value ({p_value:.4f}) >= alpha ({alpha}), "
          "therefore we fail to reject the null hypothesis.")
    print("Conclusion: There is not enough evidence to say there is a "
          "statistically significant difference in mean accuracies between TeamMixed and TeamNone.")