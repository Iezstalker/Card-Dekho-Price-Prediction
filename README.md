Predict the price of the car using ML Models(Logistic Regression & Decision Tree)

Purpose and Scope
This page identifies and explains the most important features that predict employee attrition, as discovered by the Logistic Regression and Decision Tree models. The analysis examines both model coefficients and feature importances to understand which employee characteristics most strongly influence the likelihood of leaving the organization.

For implementation details of the models themselves, see Logistic Regression Model and Decision Tree Model. For guidance on translating these insights into business actions, see Actionable Recommendations.

Feature Importance Methodology
The system employs two complementary approaches to identify attrition drivers:

Logistic Regression Coefficients
Logistic Regression produces coefficients that indicate both the direction and magnitude of each feature's impact on attrition probability. Positive coefficients increase attrition likelihood, while negative coefficients decrease it. These coefficients are extracted and analyzed at 
EmployeeAttrition.ipynb
398

Decision Tree Feature Importances
Decision Tree Classifier calculates Gini-based importance scores that measure how much each feature contributes to reducing classification uncertainty across all splits in the tree. Higher scores indicate more critical features for the decision-making process. These importances are computed at 
EmployeeAttrition.ipynb
399

Sources: 
EmployeeAttrition.ipynb
225-232
 
EmployeeAttrition.ipynb
398-402

Feature Importance Analysis Architecture






















Sources: 
EmployeeAttrition.ipynb
188-190
 
EmployeeAttrition.ipynb
225-232
 
EmployeeAttrition.ipynb
398-402

Top Attrition Drivers: Logistic Regression
The Logistic Regression model identifies features with the strongest positive coefficients as primary attrition drivers. The table below presents all features ranked by coefficient value:

Rank	Feature	Coefficient	Odds Ratio	Interpretation
1	OverTime	0.743	2.10x	Working overtime more than doubles attrition odds
2	NumCompaniesWorked	0.396	1.49x	Each additional prior employer increases odds by ~49%
3	DistanceFromHome	0.220	1.25x	Greater commute distance increases odds by ~25%
4	Education	0.039	1.04x	Minimal positive impact
5	BusinessTravel	-0.016	0.98x	Nearly neutral
6	DailyRate	-0.121	0.89x	Higher daily rate reduces odds by ~11%
7	MonthlyIncome	-0.311	0.73x	Higher monthly income reduces odds by ~27%
8	Age	-0.312	0.73x	Older employees ~27% less likely to leave
9	JobSatisfaction	-0.328	0.72x	Higher satisfaction reduces odds by ~28%
10	TotalWorkingYears	-0.347	0.71x	More experience reduces odds by ~29%
Coefficient Interpretation
Positive coefficients (increase attrition risk):

OverTime has the strongest positive coefficient, indicating employees who work overtime are at significantly elevated risk
NumCompaniesWorked shows job-hopping history is a strong predictor
DistanceFromHome suggests commute burden contributes to attrition
Negative coefficients (decrease attrition risk):

TotalWorkingYears, JobSatisfaction, and Age are the strongest protective factors
MonthlyIncome demonstrates that compensation matters, though not as much as satisfaction or experience
These features indicate employee stability and contentment
Sources: 
EmployeeAttrition.ipynb
398
 
EmployeeAttrition.ipynb
402

Top Attrition Drivers: Decision Tree
The Decision Tree model uses Gini impurity to determine which features provide the most information gain when splitting nodes. The table below shows feature importances:

Rank	Feature	Importance	Interpretation
1	MonthlyIncome	0.325	Most critical feature for tree splits (~32.5% of total)
2	OverTime	0.272	Second most important (~27.2% of total)
3	TotalWorkingYears	0.111	Significant contributor (~11.1%)
4	DailyRate	0.109	Moderate importance (~10.9%)
5	Age	0.089	Moderate importance (~8.9%)
6	NumCompaniesWorked	0.062	Minor importance (~6.2%)
7	JobSatisfaction	0.033	Minor importance (~3.3%)
8	BusinessTravel	0.000	Not used in tree splits
9	Education	0.000	Not used in tree splits
10	DistanceFromHome	0.000	Not used in tree splits
Tree-Based Interpretation
The Decision Tree prioritizes different features than Logistic Regression:

MonthlyIncome is by far the most important splitting criterion, suggesting clear income thresholds separate attrition-prone employees
OverTime remains critical in both models, confirming its universal predictive power
TotalWorkingYears and Age (experience/maturity) are valued more by the tree than individual satisfaction
Three features (BusinessTravel, Education, DistanceFromHome) are not used at all, indicating they provide redundant information given other features
Sources: 
EmployeeAttrition.ipynb
399

Model Consensus and Divergence

















Sources: 
EmployeeAttrition.ipynb
398-402

Code Implementation: Feature Analysis
The feature importance analysis is implemented in a single code cell that extracts and displays results from both trained models:

# Extract Logistic Regression coefficients
log_reg.coef_[0]  # Shape: (10,) array of coefficients

# Extract Decision Tree importances  
decision_tree.feature_importances_  # Shape: (10,) array of importances

# Calculate odds ratios from coefficients
np.exp(log_reg.coef_[0])  # Exponential transformation

# Display results in sorted DataFrames
pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
The selected_features list is defined earlier in the preprocessing phase and contains exactly 10 feature names that correspond to the indices in the coefficient and importance arrays.

Sources: 
EmployeeAttrition.ipynb
188-190
 
EmployeeAttrition.ipynb
225-227
 
EmployeeAttrition.ipynb
229-232
 
EmployeeAttrition.ipynb
398-402

Key Attrition Drivers Summary
Based on the combined analysis of both models, the following features emerge as the most critical attrition drivers:

Universal High-Risk Factors
Driver	Evidence	Business Implication
OverTime	LR coefficient: 0.743 (2.10x odds)
DT importance: 0.272 (#2)	Employees working overtime are at extreme risk and should be prioritized for retention efforts
MonthlyIncome	LR coefficient: -0.311 (0.73x odds)
DT importance: 0.325 (#1)	Compensation is the single most important decision point; clear salary thresholds exist
Logistic Regression Highlights
NumCompaniesWorked (0.396 coefficient): Job-hopping history strongly predicts future attrition
DistanceFromHome (0.220 coefficient): Long commutes create attrition risk
JobSatisfaction (-0.328 coefficient): Satisfaction is a strong protective factor
Decision Tree Highlights
TotalWorkingYears (0.111 importance): Experience creates distinct attrition risk groups
DailyRate (0.109 importance): Short-term compensation metrics matter for tree splits
Age (0.089 importance): Age-based career stages influence attrition patterns
Unused Features
The following features were excluded by the Decision Tree (0.000 importance), suggesting they are either highly correlated with other features or provide minimal unique information:

BusinessTravel
Education
DistanceFromHome
Sources: 
EmployeeAttrition.ipynb
361-394
 
EmployeeAttrition.ipynb
398-402

Feature Interaction with Attrition Target











Sources: 
EmployeeAttrition.ipynb
100-105
 
EmployeeAttrition.ipynb
361-394

Practical Application
To utilize these attrition drivers in practice:

Model Predictions: When the trained log_reg or decision_tree models generate predictions, these feature importances explain why an employee is predicted to leave or stay

Risk Scoring: Human Resources can use coefficient magnitudes to create risk scores:

Risk Score = (0.743 × OverTime) + (0.396 × NumCompaniesWorked) + ...
Intervention Prioritization: Focus retention efforts on employees with high-risk feature combinations (e.g., working overtime + low income + high companies worked)

Policy Changes: Address systemic issues revealed by top drivers (e.g., overtime policies, compensation benchmarking)

For specific business recommendations derived from these drivers, see Actionable Recommendations.
