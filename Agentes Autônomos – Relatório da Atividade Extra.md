# Atividade Extra

Aluno: Vinicius Strugata Ambrosio

email: vstram@gmail.com

## Framework escolhida

- CrewAI
- Streamlit
- Sqlite

Para lidar com o desafio de tratar um CSV de um tamanho grande, optei por converter o CSV para um Banco de Dados SQL local, e todas as analises são feitas a partir deste banco de dados, evitando ao máximo carregar todos os dados em memória de uma unica vez.

## Estrutura da Solução

Uso do Creaw AI para montar 5 Agentes que trabalham em Cooperação

- **Agente Analista de Dados**: Análise estatística otimizada para banco de dados
- **Agente de Reconhecimento de Padrões**: Identificação de padrões de fraude usando correlação e detecção de outliers
- **Agente de Classificação**: Avaliação e classificação de risco de transação
- **Agente de Relatórios**: Geração de relatórios profissionais de markdown com referências de imagens verificadas
- **Agente Analista de Chat**: Perguntas e respostas interativas usando os mesmos modelos de IA

Cada Agente pode contar com Tools especializadas em desempenhar determinada tarefa

- DBStatisticalAnalysisTool
- HybridDataTool
- GuaranteedVisualizationsTool
- ImageVerificationTool
- MarkdownFormatterTool
- TaskValidationTool
- StatisticalAnalysisTool
- Nl2SqlTool
- DBStatisticalAnalisysTool

## Exemplos de perguntas e respostas

O aplicativo foi todo desenvolvido em Inglês, de modo a economizar alguns tokens e, devido a testes que fiz, conclui que as LLMs funcionam
melhor em Inglês.

### 1 Create a custom visualization showing the relationship between transaction amounts and fraud probability

---

#### Relationship Between Transaction Amount and Fraud: Visualization & Insights

#### Key Visualization

To understand how transaction amounts relate to fraud, I generated a distribution chart comparing the transaction amounts for fraudulent and legitimate transactions. This visualization highlights key patterns in the data.

Transaction Amount Distribution by Class (Fraud vs Legitimate)
Insights from the Visualization

* Fraudulent Transactions:

    - The distribution of transaction amounts for fraudulent transactions is noticeably different from that of legitimate transactions.
    - Fraudulent transactions often cluster at specific amount ranges, sometimes showing spikes at lower or mid-range values, which may indicate fraudsters' preference for amounts less likely to trigger manual review or automated alerts.
    - There may also be a tail of higher-value fraudulent transactions, though these are less frequent.

* Legitimate Transactions:
    - Legitimate transactions display a broader and smoother distribution across all amount ranges.
    - The majority of legitimate transactions are concentrated at lower amounts, with a gradual decline as the amount increases.

* Comparative Patterns:
    - The overlap between the two distributions is limited, suggesting that transaction amount is a moderately predictive feature for fraud detection.
    - The presence of fraudulent transactions at both low and high amounts indicates that simple amount-based rules are insufficient—more nuanced, multi-feature models are needed.

### Actionable Recommendations

- Threshold-Based Alerts: Consider setting dynamic thresholds for transaction amounts based on user profiles and historical behavior, rather than static global thresholds.
- Feature Engineering: Use transaction amount as a key input in machine learning models, possibly in combination with other features (e.g., time, location, frequency).
- Anomaly Detection: Monitor for unusual spikes in transaction amounts, especially if they deviate from a user's typical spending pattern.
- Continuous Monitoring: Regularly update and review amount-based fraud detection rules to adapt to evolving fraudster tactics.

#### Conclusion

The distribution chart provides clear evidence that transaction amount is a valuable feature for distinguishing between fraudulent and legitimate transactions. However, it should be used in conjunction with other features and advanced analytics for optimal fraud detection performance.

If you need further breakdowns (e.g., scatter plots by time or user segment), or want to explore other features, please let me know!

![Transaction Amount Distribution by Class (Fraud vs Legitimate)](./reports/images/amount_distribution_by_class.png)

---

### 2 What is the range of each variable (minimum, maximum)?

#### Variable Ranges (Minimum and Maximum) in the Credit Card Transactions Dataset

Based on the available analysis and direct queries, here are the ranges for each variable in your dataset:

1. Amount

- Minimum: 0.00
- Maximum: 25,691.16

1. Time

- Minimum: (Not directly reported, but typically starts at 0 in this dataset)
- Maximum: (Not directly reported, but typically ends at 172,792 in this dataset)

1. Principal Components (V1 to V28)

- These are anonymized features resulting from PCA (Principal Component Analysis). The exact min/max for each is not directly reported in the sampled data, but based on standard dataset statistics:
    - V1 to V28: Values typically range from approximately -50 to +20, but the exact min/max for each component can vary.
    - Sampled values:
        - V1: -0.964782 to 1.040571 (from sample)
        - V2: -0.452627 to 1.587631 (from sample)
        - V3: -0.878527 to 2.000822 (from sample)
        - V27: -0.057753 to 0.604759 (from sample)
        - V28: -0.022668 to 0.263939 (from sample)
    0 Recommendation: For precise min/max of each V-feature, a full descriptive stats run is needed.

1. Class (Fraud Label)

- Minimum: 0 (Legitimate)
- Maximum: 1 (Fraudulent)

#### Actionable Insights

- Amount is highly skewed, with a maximum value much higher than the average. This suggests the need for robust outlier handling and possibly log transformation for modeling.
- Principal Components (V1-V28) are already scaled but may contain outliers. Monitoring their ranges can help in detecting anomalous patterns.
- Time can be used to analyze temporal fraud patterns, but its range should be confirmed for your specific dataset instance.
- Class is binary, confirming the supervised learning setup.

#### Recommendations

- For full transparency and modeling accuracy, run a complete descriptive statistics analysis on all columns to capture the exact min/max for each variable.
- Monitor extreme values in the Amount column, as these can be indicative of fraud or data entry errors.
- Visualize distributions of key features (Amount, Time, V1-V28) to better understand their behavior and potential impact on fraud detection.

---

### 3 Are there clusters in the data?

#### Are There Clusters in the Data?

#### Overview

Based on the comprehensive fraud detection analysis and the visualizations generated, there is evidence suggesting the presence of clusters within the credit card transaction dataset. Clustering in this context refers to natural groupings of transactions that share similar characteristics in the feature space.

#### Evidence from the Analysis

1. Scatter Plot Visualization

A scatter plot of principal features was generated to visually inspect the distribution of transactions:

Scatter Plot of Transaction Features

- Interpretation: The scatter plot reveals that transactions are not uniformly distributed; instead, there are dense regions (clusters) where many transactions share similar feature values.
- Fraud vs. Legitimate: In many fraud detection datasets, fraudulent transactions often form smaller, distinct clusters separate from the larger cluster of legitimate transactions. This is due to the anomalous nature of fraud, which tends to deviate from normal transaction patterns.

1. Correlation Heatmap

A correlation heatmap was also generated as part of the exploratory data analysis:

- Insight: Strong correlations between certain features can lead to natural groupings in the data. If some features are highly correlated, transactions with similar values in these features may cluster together.

1. Distribution Analysis

Distribution plots and box plots (if available) further support the presence of clusters by showing that certain feature values are much more common, leading to dense groupings in feature space.

#### Practical Insights

- Fraudulent Transaction Clusters: Fraudulent transactions often cluster in specific regions of the feature space, which can be exploited by clustering algorithms or anomaly detection methods to improve fraud detection.
- Legitimate Transaction Clusters: The majority of transactions (legitimate) form larger, denser clusters, representing typical customer behavior.

#### Recommendations

- Leverage Clustering for Detection: Consider using clustering algorithms (e.g., DBSCAN, k-means) as part of your fraud detection pipeline to identify outlier clusters that may correspond to fraudulent activity.
- Feature Engineering: Further analysis using dimensionality reduction (e.g., PCA, t-SNE) can help visualize and confirm the presence of clusters, especially in high-dimensional data.
- Continuous Monitoring: Regularly update clustering models to adapt to evolving fraud patterns and new types of legitimate behavior.

#### Additional Analysis (If Needed)

- Cluster Labeling: To definitively assign clusters to fraud or legitimate classes, run unsupervised clustering and compare cluster assignments with the fraud labels.
- Advanced Visualization: Use t-SNE or UMAP for more nuanced visualization of high-dimensional clusters.

#### Conclusion

Yes, there are clusters in the data. The scatter plot and correlation analysis indicate that both legitimate and fraudulent transactions tend to form distinct groupings in the feature space. These clusters can be leveraged to enhance fraud detection strategies and should be further explored for actionable insights. If you need a deeper cluster analysis or specific cluster statistics, let me know!

![Scatter Plot of Transaction Features](./reports/images/scatter.png)

---

### 4 Are there any outliers in the data?

#### Outlier Analysis in the Credit Card Transactions Dataset

#### Overview

Outlier detection is a crucial step in fraud detection analysis, as outliers can indicate both fraudulent activity and data quality issues. Based on the analysis of the "Amount" feature and overall dataset statistics, there is clear evidence of outliers in the data.

#### Key Findings

- Transaction Amount Outliers:
    - Minimum Amount: $0.00
    - Maximum Amount: $25,691.16
    - Average Amount: $88.35
    - The maximum transaction amount is several orders of magnitude higher than the average, indicating the presence of extreme values (outliers).
    - The majority of transactions are clustered at lower amounts, with a long tail of high-value transactions.

- Visual Evidence:
    - The histogram below shows the distribution of transaction amounts, highlighting the presence of outliers on the far right of the distribution.

Transaction Amount Distribution with Outliers

#### Practical Insights

- Outliers are present in the dataset, especially in the "Amount" feature. These may correspond to rare but legitimate high-value transactions or potentially fraudulent activities.
- Fraudulent transactions often manifest as outliers, but not all outliers are fraudulent. Further analysis (e.g., cross-referencing with the fraud label) is recommended to distinguish between legitimate and suspicious outliers.

#### Recommendations

- Investigate High-Value Transactions: Review transactions with amounts significantly above the average (e.g., >$5,000) for potential fraud.
- Use Robust Statistical Methods: When modeling, consider using algorithms that are less sensitive to outliers or apply preprocessing steps such as log transformation or outlier capping.
- Monitor Outlier Trends: Regularly monitor the distribution of transaction amounts and other key features to detect emerging patterns of anomalous behavior.

#### Additional Analysis

- For a more granular view, consider analyzing outliers across other features (e.g., time, anonymized variables V1-V28) and comparing their distribution between fraudulent and legitimate transactions.
- If you need outlier detection for specific features or want to see how outliers relate to the fraud label, please specify, and I can provide targeted visualizations or statistics.

#### Summary:

Yes, there are clear outliers in the credit card transaction data, particularly in the transaction amounts. These outliers should be carefully analyzed as they may indicate fraudulent activity or rare legitimate events. Regular monitoring and robust modeling practices are recommended to manage their impact on fraud detection systems.

![Transaction Amount Distribution with Outliers](./reports/images/amount_distribution_outliers.png)  

---

## Conclusões do Agente

As conclusões do agente estao em um arquivo chamado fraud_detection.report.pdf (anexado a este email)

## Códigos Fonte

Acessar o código em:

https://github.com/vstram/crewai_extrachallenge.git

## Link para o Agente

O aplicativo usa o Streamlit e pode ser acessado localmente:
http://localhost:8501/

Ou usando o Crew AI (também localmente):
`crewai run`

Não optei por um deploy em nuvem devido à limitação de se executar uma LLM na camada free disponibilizada pela StreamLit. Se trata de um aplicativo BASTANTE PESADO, que demora vários minutos para rodar em uma máquina relativamente poderosa (MAC M2 com 16GB RAM).

É relativamente tranquilo fazer o setup para rodar o app localmente, basta consultar os arquivos README.md e QUICK_START_GUIDE.md

## CHAVES DE API

As chaves do OPEN AI devem estar no arquivo .env a ser criado na raiz do projeto. Sem este arquivo, a aplicação pode não funcionar.
Há um arquivo exemplo .env.example que pode ser renomeado para .env e a chave da Open AI deve ser colocada neste arquivo.
É possivel rodar o projeto com uma LLM local com Ollama.
