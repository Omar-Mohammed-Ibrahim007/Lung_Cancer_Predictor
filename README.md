<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Lung Cancer Risk Prediction System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            font-family: "Segoe UI", Roboto, Arial, sans-serif;
            margin: 0;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.7;
        }

        header {
            background: linear-gradient(135deg, #2c7be5, #00b4d8);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }

        header h1 {
            margin: 0;
            font-size: 2.6em;
        }

        header p {
            font-size: 1.1em;
            opacity: 0.95;
        }

        .container {
            max-width: 1100px;
            margin: auto;
            padding: 30px 20px;
        }

        section {
            background: white;
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        }

        section h2 {
            border-left: 5px solid #2c7be5;
            padding-left: 12px;
            margin-top: 0;
        }

        ul {
            padding-left: 22px;
        }

        li {
            margin-bottom: 6px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        table th, table td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }

        table th {
            background-color: #2c7be5;
            color: white;
        }

        code {
            background-color: #eef1f6;
            padding: 3px 6px;
            border-radius: 5px;
            font-size: 0.95em;
        }

        pre {
            background-color: #1e1e2f;
            color: #dcdcdc;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }

        footer {
            text-align: center;
            padding: 20px;
            background-color: #1e1e2f;
            color: #ccc;
            font-size: 0.9em;
        }

        .tag {
            display: inline-block;
            background-color: #e3f2fd;
            color: #1565c0;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 4px 4px 0 0;
        }
    </style>
</head>
<body>

<header>
    <h1> Lung Cancer Risk Prediction System</h1>
    <p>Machine Learning–Based Healthcare Risk Assessment with Streamlit Deployment</p>
</header>

<div class="container">

<section>
    <h2> Project Overview</h2>
    <p>
        This project implements a complete Machine Learning pipeline for predicting lung cancer risk
        using a <strong>synthetic yet medically realistic dataset</strong>. It covers data generation,
        preprocessing, model training, hyperparameter tuning, evaluation, and deployment via a
        Streamlit web application.
    </p>
</section>

<section>
    <h2> Objectives</h2>
    <ul>
        <li>Generate realistic healthcare data with missing values and outliers</li>
        <li>Apply robust preprocessing and feature engineering</li>
        <li>Train and tune multiple classification models</li>
        <li>Prevent overfitting through validation-based selection</li>
        <li>Deploy predictions using an interactive web interface</li>
    </ul>
</section>

<section>
    <h2> Models Used</h2>
    <div class="tag">Logistic Regression</div>
    <div class="tag">KNN</div>
    <div class="tag">Decision Tree</div>
    <div class="tag">Random Forest</div>
    <div class="tag">Extra Trees</div>
    <div class="tag">XGBoost</div>
    <div class="tag">LightGBM</div>
    <div class="tag">Naive Bayes</div>
</section>

<section>
    <h2> Dataset Characteristics</h2>
    <ul>
        <li>40,000 synthetic patient records</li>
        <li>Balanced target classes (Yes / No)</li>
        <li>10% missing values</li>
        <li>2% injected outliers</li>
        <li>Label noise for realism</li>
    </ul>
</section>

<section>
    <h2> Preprocessing Pipeline</h2>
    <ul>
        <li>Data imputation using drop method to remove missing data </li>
        <li>Categorical encoding:
            <ul>
                <li>OrdinalEncoder for ordered features</li>
                <li>One-Hot Encoding for nominal features</li>
                <li>LabelEncoder for binary features</li>
            </ul>
        </li>
        <li>StandardScaler for numerical normalization</li>
        <li>All encoders and scalers saved and reused</li>
    </ul>
</section>

<section>
    <h2> Hyperparameter Tuning</h2>
    <p>
        GridSearchCV was applied to every model to optimize hyperparameters that control:
    </p>
    <ul>
        <li>Model complexity</li>
        <li>Bias–variance trade-off</li>
        <li>Overfitting and underfitting</li>
    </ul>
    <p>
        Final models were selected based on <strong>validation performance</strong>,
        not training accuracy.
    </p>
</section>

<section>
    <h2> We Choose Best Parameters Based on (Avoiding Overfitting,Higher Validation Accuracy)</h2>
    <h2> Performance Comparison 'Validation Accuracy In NoteBook!'</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Tuned Accuracy</th>
            <th>Default Accuracy</th>
        </tr>
        <tr>
            <td>Logistic Regression</td>
            <td>0.857981</td>
            <td>0.857894</td>
        </tr>
        <tr>
            <td>Random Forest</td>
            <td>0.879538</td>
            <td>0.998793</td>
        </tr>
        <tr>
            <td>XGBoost</td>
            <td>0.868500</td>
            <td>0.907045</td>
        </tr>
        <tr>
            <td>LightGBM</td>
            <td>0.871346</td>
            <td>0.871260</td>
        </tr>
    </table>
</section>

<section>
    <h2> Streamlit Web Application</h2>
    <ul>
        <li>Sidebar-based patient input</li>
        <li>Model selection dropdown</li>
        <li>Risk probability output</li>
        <li>Clean UI with cached models</li>
    </ul>

    <pre>
    <code>cd .\ML_final_Project</code>
    <code>streamlit run .\app.py</code>
    </pre>
</section>

<section>
    <h2> Disclaimer</h2>
    <p>
        This project is for academic and research purposes only.
        It is <strong>not</strong> intended for real-world medical diagnosis.
    </p>
</section>



</div>

<footer>
    © 2025 Lung Cancer Prediction Project • Machine Learning & Streamlit Deployment
</footer>

</body>
</html>
