import sys
from pathlib import Path

module_dir = Path(__file__).resolve().parent
code_dir = module_dir.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from explainability.config import (
    CLASS_TO_EXPLAIN,
    FINAL_BEST_PARAMETERS_PATH,
    RUN_NAME,
    SAMPLE_POSITION,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    create_output_directories,
)
from explainability.permutation_importance import run_permutation_importance
from explainability.plots import run_permutation_importance_plot
from explainability.shap_force import run_shap_force
from shared.modeling import load_development_and_test, rebuild_final_model


def main() -> None:
    """Run global and local explainability."""
    create_output_directories()

    X_train, y_train, w_train, X_test, y_test, w_test = load_development_and_test(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
    )

    model, model_name, dev_score, parameters = rebuild_final_model(
        X_train=X_train,
        y_train=y_train,
        sample_weight=w_train,
        parameters_path=FINAL_BEST_PARAMETERS_PATH,
    )

    print("\n" + "=" * 70)
    print("EXPLAINABILITY MODULE")
    print("=" * 70)
    print(f"Experiment: {RUN_NAME}")
    print(f"Rebuilt model: {model_name}")
    print(f"Training observations: {len(X_train)} (Mass: {w_train.sum():.0f})")
    print(f"Test observations: {len(X_test)} (Mass: {w_test.sum():.0f})")
    print(f"Input features: {X_train.shape[1]}")
    print(f"SHAP sample position: {SAMPLE_POSITION}")

    # Global permutation importance from nested CV
    importance_df = run_permutation_importance(model_name)
    run_permutation_importance_plot(importance_df)

    # Local SHAP explanation on the pure feature space
    run_shap_force(
        model=model,
        X_background=X_train,
        X_test=X_test,
        sample_position=SAMPLE_POSITION,
        class_to_explain=CLASS_TO_EXPLAIN,
    )

    print("\n" + "=" * 70)
    print("EXPLAINABILITY ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
