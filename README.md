# Classification-and-Predictive-Analytics
Classification and Predictive Analytics on the Ontario interest rates

## Project Overview 

This project builds a time-series classification model to predict the quarterly direction (Increase, Decrease, Stable) of the Ontario Underpayment Tax Interest Rate. The rate is set quarterly by the Ontario Ministry of Finance. 

--- 
## Data Selection 

Interest tax rates are the main control tool in the macroeconomics of the modern world. They  function as policy instruments to influence macroeconomic stability, regulate inflation, and guide economic growth. Governments shape borrowing plans because of them, companies adjust spending around their shifts, people rethink savings when they move. Every three months in Ontario, the Finance Ministry adjusts tax-related percentages which are the quiet markers of what the economy is facing at the moment (Ontario Ministry of Finance, 2026).


Furthermore, the dataset used in this analysis is sourced from the Ontario Data Catalogue which contains over a hundred data points of quarterly tax interest rates, such as overpayment rates (earned interest), appeal rates, and underpayment rates (interest charged).  This dataset is providing a view of quarterly interest rate behaviour from 1998 to 2026. 


For the first quarter of 2026, the mean interest rate is 7.25%, indicating a moderately high rate environment. Interest rates trend upward, suggesting a gradual tightening of fiscal conditions, potentially reflecting macroeconomic inflation control policies.

<img width="812" height="395" alt="Screenshot 2026-04-26 at 7 41 57 PM" src="https://github.com/user-attachments/assets/8e47052e-e437-4d03-b381-c392eed2ce4c" />

Most rate changes are minor, so the policy adjustments are incremental rather than abrupt. Outliers are minimal and may represent policy interventions or external economic shocks.


Taking a look at the interest tax rates over the years, it shows a bigger picture. The standard deviation is 0.42%, suggesting low short-term volatility. The minimum interest rate is 6.8%, and the maximum is 7.9%. 


Overall, the relatively small changes in rates suggest a stable policy environment with controlled adjustments over time. 


From descriptive statistics, it is known that the average interest rate is about 7.25% which is suggesting a stable policy trend with subtle adjustments and no major changes.

<img width="410" height="329" alt="Screenshot 2026-04-26 at 7 46 28 PM" src="https://github.com/user-attachments/assets/c9149105-0f5a-4eef-955a-1569ffe86579" />

This figure represents the stability of tax interest rates with a narrow range, central median, balanced distribution. No significant outliers. This boxplot is supporting a theory of a stable policy environment. 

<img width="371" height="291" alt="Screenshot 2026-04-26 at 7 47 46 PM" src="https://github.com/user-attachments/assets/a955fb0c-dcaa-47f5-8089-5c6b6b357973" />

This histogram represents low variability in time. Interest rate adjustments are gradual rather than extreme that reflect a stable and controlled policy environment. 

<img width="347" height="265" alt="Screenshot 2026-04-26 at 7 48 53 PM" src="https://github.com/user-attachments/assets/a5213fae-4683-4676-b8e5-821671b10e7f" />

This figure illustrates the temporal evolution of interest rates over time, highlighting gradual policy adjustments and periods of relative stability. The trend suggests a controlled and incremental approach to fiscal policy implementation.


Nevertheless, the dataset is somewhat limited in some aspects. For example, it does not include factors of the macroeconomics like inflation, GDP growth and other indicators that can influence the direction of interest rate changes (Stock & Watson, 2002). This is crucial since economic and policy changes are important to consider when building a predictive financial model. 


Traditional approaches usually rely on regression based models that are predicting continuous values, usually by tracking things like inflation or economic output instead of broader trends and fail to capture non-linear relationships in financial data (Stock & Watson, 2002). 


Traditional methods do not align to real-world scenarios where the direction of change is the main objective. This project, on the other hand, is focusing on sorting likely movements of interest rates into directions and predictions for a classification-based framework. 


The goal is to predict future trends, whether it will go upward, downward, or stay steady. Instead of fine-tuned forecasts, attention turns to movement patterns. A signal pointing upward might reshape how loans get handled, even without exact figures attached. The analysis is focusing on sorting likely movements of interest rates into directions and predictions. 


This approach is aligning with recent studies that are underlining the importance of directional forecasting in financial models (Salem & Albourawi, 2024). Predictive analytics is more practical and relatable to real problems in the lens of directional forecasting (Sokolova & Lapalme, 2009).


The goal of the analysis is to see if the machine learning tools can handle direction-based movement sorting well, while also digging into what shapes those forecast outcomes. This work feeds into real-world issues like shaping policy choices or guiding budget strategies. 


Research is looking into questions whether machine learning is capable of classifying quarterly interest rate movements, how feature engineering is affecting model performance. Also, the research is evaluating if models remain stable over time or change, and which features are influencing the predictions the most. Those and many more aspects are discussed in the analysis of the interest rate trajectories of this project.


Each step rests on what came before, forming layers instead of standing alone. Their shared goal isn’t just measuring correctness but exploring the reasons behind model behaviour. Progress comes not from isolated answers but from linked insights that reveal underlying patterns.

---
## Research Questions

Both predictive capacity and machine learning in models of Ontario interest rates are evaluated. Whether machine learning models can effectively classify the direction of quarterly interest rates is the first and the main question to explore. It is foundational since a model is supposed to show that it is more effective than traditional alternative, regression-based models. 

The second question is looking into the major contributions of feature engineering, temporal features such as lag variables and rolling statistics. Historic dependencies could be improving predictive models and determining any possible patterns that could be useful for future predictive modeling. 

It is crucial to investigate feature importance and interpretability. The study investigates the impact of temporal feature engineering on predictive performance. Feature engineering is widely recognized as a critical component of machine learning, as it transforms raw data into meaningful inputs for models (Domino Data Lab, n.d.).

Then, the study compares the performance of different machine learning models, including Logistic Regression, Random Forest and XGBoost. This is a check if the more complex models provide significant improvements due to their ability to capture nonlinear relationships (Chen & Guestrin, 2016).


Another research question is looking into the stability of the model. The model should be accurate for any period of time. This matter can be checked through cross-validation testing in order to make sure the model is robust (Bergmeir, & Benítez, 2012).


All questions together form a picture that is showing more beyond predictions and more into deeper understanding of patterns and insights of how changes occur and what actually affects them the most. 


The study is showing multiple concepts for a classification approach to interest rate predictions. It focuses on temporal feature engineering, incorporating rolling statistics and lag variables. It depicts a classification that aligns with time-series forecasting theory (Hyndman & Athanasopoulos, 2021). Also, the research captures a full framework of effectiveness, efficiency, and stability. 


Interest rate environments are not static, they evolve in response to many macroeconomic shocks, policy cycles, or financial crises. Therefore, it is quite important to assess whether models trained on historical data maintain predictive validity whenever structural breaks occur. This aligns with concerns in time-series modeling where non-stationarity can degrade predictive accuracy (Hyndman & Athanasopoulos, 2021).


Additionally, the study can also explore feature stability over time. Features such as lag variables and rolling statistics may vary in importance depending on economic conditions. Measuring feature importance dynamically across time windows would provide insights into whether predictive drivers remain consistent or are shifting across periods (Guyon & Elisseeff, 2003). 


In conclusion, important research questions focus on decision-making ability rather than pure predicting accuracy. Even if a model achieves high classification performance, its real-world value depends on how effectively predictions inform financial or policy decisions in the long run (Varian, 2014).

---
## Proposed Methodology and Tools                                                                                                                                                                                                                                                           
Data processing is greatly impacting model performance. The process includes data preprocessing, feature engineering, model development, and evaluation. 
First, the dataset was cleaned and all of the inconsistencies, missing values, and formatting issues were removed. This is done because machine learning models are very sensitive to the quality of the data.                           
Then, the data was sorted chronologically since the dataset is a time-series dataset and temporal structure has to stay intact. Using the first difference of the dataset, the target variable has been done. This transformation has converted a continuous variable into a categorical value. 


The most important step of the research is feature engineering. Lag variables are used for temporal dependencies. lag_1 represents a previous quarter and lag_2 represents two quarters prior. So, those features are making the model remember values from past quarters to use for the future predictions. 

<img width="615" height="512" alt="Screenshot 2026-04-26 at 7 52 25 PM" src="https://github.com/user-attachments/assets/0e795465-8cce-4b5a-b6ec-ebbe817a8787" />

The heatmap provides insight into the relationships between variables, helping to identify potential multicollinearity and inform feature selection.


Rolling statistics is showing that the rolling mean is capturing short-term trends and rolling standard deviation is capturing volatility. These features are stabilizing future predictions. To test performance on unseen data, 80% of the data used for training and 20% reserved for evaluation. Training data was split into many subsets and results are averaged for reliable performance. 


Methodology demonstrates a framework that aligns with data acquisition, feature engineering, and model development into one pipeline. This approach minimizes variance and enhances scalability. Model decisions are evidence based which is strengthening robustness and reliability. 

---
## Model Evaluation 


The effectiveness of the model was measured using accuracy, precision, recall, and F1 score. In this study, two primary models were developed: Logistic Regression, Random Forest and XGBoost. Logistic Regression was used as a baseline model due to its simplicity and interpretability. Random Fores captured non-linear relationships and feature interactions.
XGBoos shows high performance and ability to model complex patterns. The dataset was also split into training and testing sets to evaluate model performance.


Logistic Regression is a base model because of its simplicity and interpretability. It is modeling the probability of the target variable as the function of inputted features, assuming linear relationship. 


<img width="539" height="406" alt="Screenshot 2026-04-26 at 7 54 27 PM" src="https://github.com/user-attachments/assets/6c14f4dc-0092-40bf-935b-acd232e48c4e" />

This matrix presents the classification performance of the Logistic Regression model, showing correct and incorrect predictions across both classes.


Random Forest is combining multiple decision trees. Each tree is trained on the subset of the data and the final prediction is made by gathering all of the trees. 

<img width="475" height="455" alt="Screenshot 2026-04-26 at 7 55 17 PM" src="https://github.com/user-attachments/assets/2444b83b-6f1a-4582-a2ca-3b3a02b228ff" />

This figure illustrates the improved classification performance of the Random Forest model compared to Logistic Regression. It reveals a model that is well performed and identifies both majority and minority classes.

Moreover, when the XGBoost model is performed, it is able to deal perfectly with complex patterns. The results are showing the highest accuracy, highest F1 score, and improved recall for minority classes.

<img width="378" height="363" alt="Screenshot 2026-04-26 at 7 56 04 PM" src="https://github.com/user-attachments/assets/018d68f3-1ffb-43de-9b5f-9d288592cbb9" />

The confusion matrices provide a thorough and detailed view of model performance, showing the amount of correct and incorrect predictions for each class.

So, accuracy measures overall correctness. Precision measures correctness of positive predictions. Recall measures ability to detect actual positives. F1 Score balances precision and recall. Cross-validation was also applied to ensure model stability and reliability.

<img width="440" height="345" alt="Screenshot 2026-04-26 at 7 57 11 PM" src="https://github.com/user-attachments/assets/22d15795-b63f-4469-9e77-5917353c3cbd" />

Precision remains high even when recall increases, suggesting that accuracy is maintained when detecting increases in tax interest rates. 

<img width="474" height="358" alt="Screenshot 2026-04-26 at 7 58 05 PM" src="https://github.com/user-attachments/assets/3ccf1dda-4dff-4e5b-9eca-162819d49b36" />

Model stability across multiple folds, demonstrating the robustness and reliability of predictions.

<img width="506" height="390" alt="Screenshot 2026-04-26 at 7 59 17 PM" src="https://github.com/user-attachments/assets/5077370c-76dc-4e94-88bd-ed600543c966" />

This bar chart compares the overall accuracy of different models, demonstrating the superior performance of ensemble methods.

<img width="489" height="379" alt="Screenshot 2026-04-26 at 8 01 16 PM" src="https://github.com/user-attachments/assets/ed2a2f91-83a8-466a-b4e8-a5cf74efbf77" />

Therefore, it is shown that Random Forest significantly outperforms Logistic Regression and XGBoost across multiple measures, particularly in terms of recall and F1 score. So, correct evaluation metrics and ensemble methods are able to deal with nonlinear relationships very well. 

Throughout the use of cross-validation and various comparisons, it is transparent that the evaluation framework is effectively representing both strong and weak risk areas. Key indicators such as accuracy, precision, and recall are great at showing competitive performance. The evaluation is validating the model's ability to show its effectiveness. 

---

## Findings and Interpretations

The results of the study demonstrate significant evidence that machine learning models are capable of effectively predicting the directional movement of Ontario tax interest rates. Logistic Regression, Random Forest, and XGBoost models have shown great levels of accuracy. 
One thing to note from those models is that some differences are visible when  evaluating recall and F1 score. Overall, classification problems that have class imbalance would require correct evaluation criteria because of the differences that are present in the models. 

The Logistic Regression model has great accuracy but also some limitations when identifying instances of increasing interest rates. It is noticeable because of the low recall for the positive class, so the model is not capturing all of the upward rate movements. This happens because Logistic Regression is a linear model and, therefore, can be limited in representing complex and nonlinear relationships in the data. Since the dataset of the study is a time-series data and interactions between variables are nonliners, limitations are occurring in this model and affect results of the predictive performance. 

On the other hand, the Random Forest model is representing a stronger performance within all of the evaluation metrics. Recall and F1 score is notably larger, this is indicating more precise and accurate classification of the both positive and negative classes. Random Forest is collecting predictions from multiple decision trees to reduce overfitting and detect complex patterns of the dataset. Because Random Forest is able to model all of the interactions between features of nonlinear relationships, it is suitable for the complex financial time-series dataset and predicts accurate outcomes. 

Also, the XGBoost model was tested alongside the other models. It is able to deal with complex datasets, just as Random Forest but the XGBoost is building trees sequentially, where each tree is designed to correct mistakes of the previous tree. This strategy may result in higher predictive or similar performance to Random Forest since they are similar and both have proven to be effective.

Comparing results from all three models, it is useful to look at the accuracy numbers. 
Logistic Regression Accuracy is 0.8695652173913043. Random Forest Accuracy is 1.0000000000000000. And XGBoost Accuracy is 0.9565217391304348.

Therefore, both XGBoost and Random Forest have highest accuracy scores but for this dataset, Random Forest is the perfect model with ideal score. 

There are more insights in interest rate behaviour that are noticeable during feature engineering. The analysis is showing that lag features, especially lag_1 feature, which is representing the previous quarter, is significant in predicting future interest tax rate movements. This suggests that historical data is strongly influencing future interest tax rate movements which is supporting the hypothesis of temporal dependency. 

An economic theory is claiming that policymakers usually rely on recent data during the decision making process and the observation from the research is supporting this theory.  Current economic conditions such as inflation or economic growth are affecting interest tax rate adjustments. In conclusion, lag variables are capable of predicting the results of the policies based on historical data. 

Looking closer into the details of the analysis, predictive power can also be attributed to the rolling statistical features. Rolling mean and rolling standard deviation are representing short-term trends and providing more variability for the model. They suggest that both the level and volatility of interest rates are significant factors to predict future adjustments. 

Not to mention, a high level of stability is important for the analysis. It is reflected in the distribution of the rate changes that is mostly concentrated around zero. So, the classification problem is unbalanced because the majority of adjustments have no change or very minor adjustments. This can be observed in Figure 1 of the report. The imbalance is not great for the model evaluation and highlights the importance of evaluation metrics such as F1 score that is responsible for both precision and recall. 

The results of the study depict a picture that the direction of the interest tax rates has a pattern that is useful for policymakers, businesses, and investors. A shift like an increase in interest tax rate can inform many decisions related to borrowing, investments and, overall risk management of the upcoming time. 

Policy behaviour can be analyzed because of the machine learning models. They are accurate at identifying consistent patterns and relationships within the data. Due to the accuracy of the models, the decision making process can be more efficient and backed by data. Complex systems become a replacement for traditional methods because of the precision of machine learning models in recent years. 

In conclusion, the results of the research are representing approaches that are learning and predicting tax interest rate behavior over time. Since the financial datasets are mostly nonlinear, the technique of modeling should be able to deal with this in order to be correct and not miss any important information. 

Temporal feature engineering is a key of the process since it is using lag features that drive main predictions in the model. Also, appropriate evaluation metrics are crucial for the best outcomes. For the full picture of the presdictions, it is necessary to look at the broader economic context. 

---
## Limitations 

Even though the machine learning models are providing accurate and detailed results, it is still useful to think of the limitations that might arise that can affect the results of the findings. 

One of the most important things to consider is the size of the dataset. The span of the dataset is about 28 years and it consists of quarterly observations which is actually resulting in a limited amount of data points. For the machine learning models, bigger datasets perform usually better since the overfitting is avoided. Larger datasets are providing models with more robust patterns overall. 

Another limitation of the research dataset is the absence of macroeconomic variables. Interest rates are usually being influenced by many factors such as inflation, GDP growth, unemployment rate, and changes in monetary policies. The absence of those factors in the dataset is limiting the ability to understand the full picture of the interest rate fluctuations. Because of this, the models are solely reliant on historical patterns which do not fully portray the complexity of the quarterly adjustments.

Looking further, the use of binary classification frameworks is highly simplifiable for the prediction problem. Due to the magnitude of changes, loss of information is possible to appear. This limitation is about the size of  the change which is almost as important as the direction of the change.  If the dataset is not accurately representing relevant conditions, bias may occur or even misleading predictions. 


---
## Ethics 

From the ethical standpoint, the risk of over relying on automated machine learning models for a final decision is not ideal. While models can provide lots of useful tools and insights, there should still be caution involved to consider possible mistakes and limitations that models possibly might face. The sole reliance on automated prediction can be somewhat dangerous and negligent. 

The dataset of the research is publicly available to anyone and does not contain any personal information so there is no privacy concern. Nevertheless, other datasets might contain sensitive information like geolocations, names of businesses, individuals and other data that should be protected by implementing privacy regulations and following legal protocols. 

Looking at real-world applications, the impacts of the models should be looked at. Incorrect predictions can be followed by big monetary losses for government, businesses, and individuals. Therefore, models have to be tested vigorously on specific datasets before releasing it for the public to use. 

Overall, caution always has to be involved while dealing with automated models. Ethical considerations are personal but they might affect many people in the end, if thorough thinking and double checking was not done. 

Any type of technological aspect of human life should always add to the existing system and not fully replace it, especially if errors and limitations are possible. Machine learning models are a great and useful tool in the right hands that are using them effectively and responsibly. 

---
## Future Work and Recommendations

The results of the study are opening several thoughts about future research and further developments. This analysis is working on potential machine learning rate directions and there are so many enhancements to this topic that are possible have been discussed and not discussed in this report. 

Since the research is fully evolved around one selected dataset, future work can focus on more dataset or choose one where more variables are present. Macroeconomic factors in the dataset will bring the  model to a new level. The accuracy will improve significantly if factors like inflation, GDP growth, and unemployment rate are included in the dataset (Stock & Watson, 2002).

More advanced machine learning techniques can also be explored in future projects. Even if Random Forest is showing good performance but focus can shift to more advanced models like Gradient Boosting to add more accuracy. 

Another recommendation is the adoption of advanced deep learning architectures, such as Long Short-Term Memory, LSTM networks. These models are designed for sequential data and can capture long-term dependencies more effectively than traditional machine learning approaches. Applying LSTM models can improve performance in capturing complex temporal patterns (Hochreiter & Schmidhuber, 1997).

Additionally, implementation of walk-forward validation frameworks would greatly strengthen model reliability. This approach ensures that training always precedes testing chronologically, notably reducing the risk of data leakage and improving real-world applicability. It allows for continuous model updating that is so important in dynamic financial environments (Hyndman & Athanasopoulos, 2021).

Not to mention, there is a huge potential in expanding the framework toward multi-class or regression-based hybrid models. Instead of limiting predictions to directional movement, future systems can simultaneously predict both direction and magnitude. This would actually provide more insights and enhance the practical value of the model for financial planning and risk management.

---
## References

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. Information Sciences, 191, 192–213.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. 

Domino Data Lab. (n.d.). Feature engineering overview.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223–2273.

Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157–1182.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.

Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and practice (3rd ed.). OTexts.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 4765–4774.

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427–437.

Stock, J. H., & Watson, M. W. (2002). Introduction to econometrics. Pearson.

Varian, H. R. (2014). Big data: New tricks for econometrics. Journal of Economic Perspectives, 28(2), 3–28.
