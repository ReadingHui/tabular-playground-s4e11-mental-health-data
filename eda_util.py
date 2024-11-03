import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataImport:
    def get_train_test(path: str, index_col: int = None, target: str = None, test_size: float = 0.1, random_state: int = 1048576, verbose: int = 1) -> tuple[pd.DataFrame]:
        train_csv = pd.read_csv(path, index_col=index_col)
        X, y = feature_target.feature_target_split(train_csv, target=target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        if verbose > 0:
            print(f'Shape of X_train is: {X_train.shape}; shape of y_train is: {y_train.shape}')
            print(f'Shape of X_test is: {X_test.shape}; shape of y_test is: {y_test.shape}')
        return X_train, X_test, y_train, y_test

class feature_target:
    def feature_target_split(df: pd.DataFrame, target: str, col_drop=[]):
        y = df[target]
        mask = col_drop + [target]
        if col_drop:
            X = df.drop(columns=mask, axis=1)
        else:
            X = df.drop(columns=target, axis=1)
        return X, y

class plots:
    def distributionPlots(df:pd.DataFrame, drop: str | list[str]=None) -> None:
        # Set the Seaborn style
        sns.set(style="whitegrid")
        columns = df.columns.to_list()
        if drop:
            if type(drop) == str:
                if drop not in columns:
                    raise KeyError(f'{drop} is not in the feature list.')
                else:
                    columns.remove(drop)
            elif type(drop) == list:
                for f in drop:
                    if f not in columns:
                        raise KeyError(f'{f} is not in the feature list.')
                    else:
                        columns.remove(f)

        # Define the plot size and the number of rows and columns in the grid
        num_plots = len(columns)
        rows = (num_plots + 1) // 2  # Calculate the number of rows needed (two plots per row)
        cols = 2  # Two plots per row
        _, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

        # Iterate through the numerical features and create the density plots
        for i, feature_name in enumerate(columns):
            row_idx, col_idx = divmod(i, cols)  # Calculate the current row and column index
            if df[feature_name].dtype == 'int32' or df[feature_name].dtype == 'int64' or pd.api.types.is_object_dtype(df[feature_name]):
                if df[feature_name].nunique() > 20:
                    sns.countplot(df, x=feature_name, ax=axes[row_idx, col_idx], color='darkcyan', order=df[feature_name].value_counts().iloc[:20].index)
                else:
                    sns.countplot(df, x=feature_name, ax=axes[row_idx, col_idx], color='darkcyan')
                axes[row_idx, col_idx].set_title(f'Count Plot of {feature_name}')
                axes[row_idx, col_idx].set_xlabel(feature_name)
                axes[row_idx, col_idx].set_ylabel('Count')
                axes[row_idx, col_idx].bar_label(axes[row_idx, col_idx].containers[0])
            else:
                sns.histplot(df, x=feature_name, kde=True, ax=axes[row_idx, col_idx])
                axes[row_idx, col_idx].set_title(f'Density Plot of {feature_name}')
                axes[row_idx, col_idx].set_xlabel(feature_name)
                axes[row_idx, col_idx].set_ylabel('Density')
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots

        plt.show()

class correlations:
    def featuresCorr(df: pd.DataFrame, num_features: list) -> None:
        df = df[num_features]
        cmap = sns.color_palette("light:b", as_cmap=True)
        sns.heatmap(df.corr().abs(), cmap=cmap,
                square=True, linewidths=.5, annot=True)
        plt.show()

    def pairwiseCorr(data:pd.DataFrame, target: str) -> None:
        df = data.drop(target, axis=1)
        num_features = df.columns[(df.dtypes == 'int64') | (df.dtypes == 'float64')].to_list()
        cat_features = df.columns[df.dtypes == 'object'].to_list()
        features = num_features + cat_features

        # Set the Seaborn style
        sns.set(style="whitegrid")

        # Define the plot size and the number of rows and columns in the grid
        num_plots = len(features) * (len(features) - 1) // 2
        cols = 3  # 3 plots per row
        rows = num_plots // cols + 1  # Calculate the number of rows needed (3 plots per row)
        _, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

        # Iterate through the numerical features and create the density plots
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                row_idx, col_idx = divmod(i * (2 * len(features) - i - 1) // 2 + j - i - 1, cols)  # Calculate the current row and column index
                if features[i] in num_features and features[j] in num_features:
                    sns.scatterplot(data=data, x=features[i], y=features[j], ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Scatter Plot of {features[i]} against {features[j]}')
                    axes[row_idx, col_idx].set_xlabel(features[i])
                    axes[row_idx, col_idx].set_ylabel(features[j])
                elif features[i] in num_features and features[j] in cat_features:
                    sns.countplot(data=data, x=features[j], hue=features[i], ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Count Plot of {features[j]} subject to {features[i]}')
                    axes[row_idx, col_idx].set_xlabel(features[j])
                    axes[row_idx, col_idx].set_ylabel('Count')
                elif features[j] in num_features and features[i] in cat_features:
                    sns.countplot(data=data, x=features[i], hue=features[j], ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Count Plot of {features[i]} subject to {features[j]}')
                    axes[row_idx, col_idx].set_xlabel(features[i])
                    axes[row_idx, col_idx].set_ylabel('Count')
                else:
                    sns.countplot(data=data, x=features[i], hue=features[j], ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Count Plot of {features[i]} subject to {features[j]}')
                    axes[row_idx, col_idx].set_xlabel(features[i])
                    axes[row_idx, col_idx].set_ylabel('Count')
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots

        plt.show()
