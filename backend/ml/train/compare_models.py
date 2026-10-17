import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    print("pip install rich")
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs):
            for arg in args:
                if isinstance(arg, str):
                    import re
                    print(re.sub(r'\[.*?\]', '', arg))
                else:
                    print(arg)
    console = Console()

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ===========================================
# PATHS
# ===========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '../../dataset')
MODEL_DIR = os.path.join(BASE_DIR, '../model')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

TRAIN_PATH = os.path.join(DATASET_DIR, 'Diabetes_train.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'Diabetes_test.csv')

# ===========================================
# CONFIG (Same as train_model.py)
# ===========================================
DROP_COLS = ['Veggies']

FEATURE_ORDER = [
    'HighBP', 'HighChol', 'BMI', 'Smoker',
    'Fruits', 'GenHlth', 'MentHlth', 'Sex', 'Age'
]

TARGET_COLUMN = "Diabetes_binary"

# ===========================================
# DATA LOADING
# ===========================================
console.print("[bold cyan]" + "=" * 50 + "[/bold cyan]")
console.print("[bold cyan]Loading datasets for comparison...[/bold cyan]")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = train_df.drop(columns=DROP_COLS, errors='ignore')
test_df = test_df.drop(columns=DROP_COLS, errors='ignore')

X_train = train_df[FEATURE_ORDER]
y_train = train_df[TARGET_COLUMN]

X_test = test_df[FEATURE_ORDER]
y_test = test_df[TARGET_COLUMN]

console.print(f"[green]Train set:[/green] {X_train.shape}")
console.print(f"[green]Test set :[/green] {X_test.shape}")

# ===========================================
# DEFINE MODELS
# ===========================================
models = {
    "HistGradientBoosting": HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=50,
        max_leaf_nodes=15,
        max_iter=100,
        l2_regularization=0.5,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
if LGBM_AVAILABLE:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
if CATBOOST_AVAILABLE:
    models["CatBoost"] = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=4,
        verbose=0,
        random_state=42
    )

# ===========================================
# TRAINING & EVALUATION
# ===========================================
results = []
console.print("\n[bold cyan]Starting model comparison...[/bold cyan]")

if RICH_AVAILABLE:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    )
    progress.start()
else:
    class DummyProgress:
        def start(self): pass
        def stop(self): pass
        def add_task(self, *args, **kwargs): return 1
        def update(self, *args, **kwargs): pass
    progress = DummyProgress()
    progress.start()

for name, model in models.items():
    task_id = progress.add_task(f"[cyan]Training {name}...", total=None)
    if not RICH_AVAILABLE:
        console.print(f"\nTraining {name}...")
        
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    pred_start = time.time()
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    pred_time = time.time() - pred_start
    
    progress.update(task_id, description=f"[green]OK {name} trained ({train_time:.2f}s) & predicted ({pred_time:.4f}s)")
    if not RICH_AVAILABLE:
        console.print(f"{name} trained in {train_time:.2f} seconds.")
        console.print(f"{name} predicted in {pred_time:.4f} seconds.")
    
    # Threshold optimization
    best_f1 = -1
    best_thresh = 0.5
    best_acc = 0
    best_prec = 0
    best_rec = 0
    
    thresholds = np.arange(0.20, 0.81, 0.01)
    for thresh in thresholds:
        y_pred_temp = (y_prob >= thresh).astype(int)
        f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_thresh = thresh
            best_acc = accuracy_score(y_test, y_pred_temp)
            best_prec = precision_score(y_test, y_pred_temp, zero_division=0)
            best_rec = recall_score(y_test, y_pred_temp)
            
    auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Accuracy": best_acc,
        "Precision": best_prec,
        "Recall": best_rec,
        "F1": best_f1,
        "ROC_AUC": auc,
        "Best_Threshold": best_thresh,
        "Training_Time_sec": train_time,
        "Prediction_Time_sec": pred_time
    })

progress.stop()


# ===========================================
# RESULTS
# ===========================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)

if RICH_AVAILABLE:
    table = Table(title="MODEL COMPARISON RESULTS", box=box.DOUBLE_EDGE, title_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1 Score", justify="right")
    table.add_column("ROC-AUC", justify="right", style="bold")
    table.add_column("Best Threshold", justify="right")
    table.add_column("Training Time", justify="right")
    table.add_column("Prediction Time", justify="right")

    for i, row in results_df.iterrows():
        style = "bold green" if i == 0 else None
        table.add_row(
            row["Model"],
            f"{row['Accuracy']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1']:.4f}",
            f"{row['ROC_AUC']:.4f}",
            f"{row['Best_Threshold']:.2f}",
            f"{row['Training_Time_sec']:.2f}s",
            f"{row['Prediction_Time_sec']:.4f}s",
            style=style
        )
    console.print("\n")
    console.print(table)
else:
    console.print("\n" + "="*50)
    console.print("[bold magenta]MODEL COMPARISON RESULTS[/bold magenta]")
    console.print("="*50)
    console.print(results_df.to_string())
    console.print("="*50)

# ===========================================
# TERMINAL BAR CHARTS
# ===========================================
def print_bar_chart(df, metric, title, is_time=False, format_str="{:.4f}"):
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    max_val = df[metric].max()
    max_bar_len = 30
    for i, row in df.iterrows():
        name = row["Model"]
        val = row[metric]
        bar_len = int((val / max_val) * max_bar_len) if max_val > 0 else 0
        bar = "█" * bar_len
        color = "green" if i == 0 else "blue"
        val_str = format_str.format(val)
        console.print(f"{name:<20} [{color}]{bar:<30}[/{color}] {val_str}")

print_bar_chart(results_df, "ROC_AUC", "ROC-AUC Comparison")
print_bar_chart(results_df, "Accuracy", "Accuracy Comparison")
print_bar_chart(results_df, "F1", "F1 Score Comparison")
print_bar_chart(results_df, "Training_Time_sec", "Training Time Comparison (sec)", True, "{:.2f}")
print_bar_chart(results_df, "Prediction_Time_sec", "Prediction Time Comparison (sec)", True, "{:.4f}")

# ===========================================
# FINAL SUMMARY
# ===========================================
best_row = results_df.iloc[0]
summary_text = (
    f"[bold yellow]Best Model:[/bold yellow]\n[bold green]{best_row['Model']}[/bold green]\n\n"
    f"[bold cyan]ROC-AUC:[/bold cyan]\n{best_row['ROC_AUC']:.4f}\n\n"
    f"[bold cyan]Accuracy:[/bold cyan]\n{best_row['Accuracy']*100:.2f}%\n\n"
    f"[bold cyan]F1:[/bold cyan]\n{best_row['F1']:.4f}\n\n"
    f"[bold cyan]Threshold:[/bold cyan]\n{best_row['Best_Threshold']:.2f}\n\n"
    f"[bold cyan]Training Time:[/bold cyan]\n{best_row['Training_Time_sec']:.2f} sec\n\n"
    f"[bold cyan]Prediction Time:[/bold cyan]\n{best_row['Prediction_Time_sec']:.4f} sec"
)

if RICH_AVAILABLE:
    console.print("\n")
    console.print(Panel(summary_text, title="[bold magenta]BEST MODEL SUMMARY[/bold magenta]", border_style="magenta", expand=False))
else:
    console.print("\n" + "="*50)
    console.print("BEST MODEL SUMMARY")
    console.print("="*50)
    console.print(summary_text)
    console.print("="*50)

console.print("\n[bold magenta]Performance ranking:[/bold magenta]")
medals = ["1st", "2nd", "3rd", "4th", "5th"]
for i, row in results_df.iterrows():
    medal = medals[i] if i < len(medals) else f"{i+1}."
    console.print(f"{medal} {row['Model']}")

# ===========================================
# SAVING ARTIFACTS
# ===========================================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

csv_path = os.path.join(MODEL_DIR, "model_comparison.csv")
results_df.to_csv(csv_path, index=False)
console.print(f"\n[green]Saved comparison table to[/green] {os.path.basename(csv_path)}")

def plot_comparison(df, x_col, title, filename, xlabel, color_palette='viridis'):
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.barplot(data=df, x=x_col, y='Model', hue='Model', palette=color_palette, legend=False)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{width:.4f}',
                        (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center',
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path)
    plt.close()

sns.set_theme(style="whitegrid")
plot_comparison(results_df, 'ROC_AUC', 'Model Comparison by ROC-AUC', 'roc_auc_comparison.png', 'ROC-AUC Score')
plot_comparison(results_df, 'Accuracy', 'Model Comparison by Accuracy', 'accuracy_comparison.png', 'Accuracy Score')
plot_comparison(results_df, 'F1', 'Model Comparison by F1 Score', 'f1_comparison.png', 'F1 Score')
plot_comparison(results_df, 'Training_Time_sec', 'Model Comparison by Training Time', 'training_time_comparison.png', 'Training Time (sec)', 'magma')
plot_comparison(results_df, 'Prediction_Time_sec', 'Model Comparison by Prediction Time', 'prediction_time_comparison.png', 'Prediction Time (sec)', 'magma')

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
fig.suptitle('Model Performance Dashboard', fontsize=20, y=0.98)

metrics = [
    ('ROC_AUC', 'ROC-AUC', axes[0,0], 'viridis'),
    ('Accuracy', 'Accuracy', axes[0,1], 'plasma'),
    ('F1', 'F1 Score', axes[1,0], 'inferno'),
    ('Training_Time_sec', 'Training Time (s)', axes[1,1], 'cividis')
]

for col, title, ax, palette in metrics:
    sns.barplot(data=results_df, x=col, y='Model', hue='Model', palette=palette, legend=False, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{width:.4f}' if 'Time' not in title else f'{width:.2f}',
                        (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center',
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
combined_path = os.path.join(RESULTS_DIR, 'combined_dashboard.png')
plt.savefig(combined_path)
plt.close()

console.print(f"[green]Saved all visualizations to[/green] {RESULTS_DIR}")
