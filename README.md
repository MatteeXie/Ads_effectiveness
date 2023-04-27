## Proposal of machine learning applied in advertising effectiveness
<br><br>
- Goals：
    - Looking for important factors affecting advertising effectiveness
    - Increase the accuracy of predicting advertising effectiveness
    - Research the correlations between advertising factors and effectiveness
    - Research the differences between concrete factors and abstract factors in the impact on advertising effctiveness
<br><br>
- Methods:
    - Dataset:
        - Data collection：206 ads from Amazon
        - Data cleaning: 
             1. Remove duplicate samples
             2. Data of Factors obtained by manual in the same sample might have difference
             3. Remove unnecessary factors. (eg. there are too few statistics)
             4. Process highly correlated factors
        - Statistic analysis
        - Divide the dataset: Training dataset, Validation dataset and Test dataset
    - Feature representation of ads:
        - Manual annotated 62 factors about 6 categories
    - ML methods:
        1. Multiple linear regression: Use all factors and select 3 best factors
        2. Support vector machine regression
        3. Decision tree regression
        4. Random forest regression
        5. k-fold cross validation
        6. Grid search
    - Performance evaluation:
        1. Mean square error(MSE)
        2. Root mean square error(RMSE)
        3. R-Squared: value range（-∞，1], the closer value to 1,the better the fittness of the model
<br><br>
- Expected results:
    1. Train models with all 64 factors，observe the result of prediction
    2. Train the factors that we supposed are important and the factors are not respectively，analyze whether the results of prediction are different
    3. Choose the factors that performed well in step 2，and train them individually or in groups. Ultimately, looking for the factors that will have the greatest impact on the predictive results
    4. Tree-based models can be visualized and easily understood and interpreted. We also can find the predictive rules from this model
