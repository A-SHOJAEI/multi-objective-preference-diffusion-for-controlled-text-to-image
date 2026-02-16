"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzer for training and evaluation results.

    Args:
        results_dir: Directory to save analysis outputs
    """

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_plot: bool = True,
    ) -> Dict[str, Any]:
        """Analyze and visualize training history.

        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            save_plot: Whether to save plot to file

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "num_epochs": len(train_losses),
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "final_val_loss": val_losses[-1] if val_losses else 0.0,
            "best_val_loss": min(val_losses) if val_losses else 0.0,
            "best_epoch": int(np.argmin(val_losses)) if val_losses else 0,
        }

        # Check for overfitting
        if len(train_losses) > 5 and len(val_losses) > 5:
            recent_train = np.mean(train_losses[-5:])
            recent_val = np.mean(val_losses[-5:])
            analysis["overfitting_gap"] = float(recent_val - recent_train)
            analysis["is_overfitting"] = recent_val > recent_train * 1.2

        if save_plot:
            self._plot_training_curves(train_losses, val_losses)

        return analysis

    def _plot_training_curves(
        self, train_losses: List[float], val_losses: List[float]
    ) -> None:
        """Plot training and validation curves.

        Args:
            train_losses: Training losses
            val_losses: Validation losses
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plot_path = self.results_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Training curves saved to {plot_path}")

    def analyze_preference_objectives(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        objective_names: List[str],
        save_plot: bool = True,
    ) -> Dict[str, Any]:
        """Analyze per-objective performance.

        Args:
            predictions: Predicted scores [B, num_objectives]
            targets: Target scores [B, num_objectives]
            objective_names: Names of objectives
            save_plot: Whether to save plots

        Returns:
            Dictionary with per-objective analysis
        """
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        analysis = {}

        for i, name in enumerate(objective_names):
            pred = predictions_np[:, i]
            target = targets_np[:, i]

            analysis[name] = {
                "mse": float(np.mean((pred - target) ** 2)),
                "mae": float(np.mean(np.abs(pred - target))),
                "correlation": float(np.corrcoef(pred, target)[0, 1]),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred)),
                "target_mean": float(np.mean(target)),
                "target_std": float(np.std(target)),
            }

        if save_plot:
            self._plot_objective_comparison(
                predictions_np, targets_np, objective_names
            )

        return analysis

    def _plot_objective_comparison(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        objective_names: List[str],
    ) -> None:
        """Plot comparison of predictions vs targets for each objective.

        Args:
            predictions: Predicted scores
            targets: Target scores
            objective_names: Names of objectives
        """
        num_objectives = len(objective_names)
        fig, axes = plt.subplots(1, num_objectives, figsize=(5 * num_objectives, 4))

        if num_objectives == 1:
            axes = [axes]

        for i, (ax, name) in enumerate(zip(axes, objective_names)):
            pred = predictions[:, i]
            target = targets[:, i]

            ax.scatter(target, pred, alpha=0.5, s=20)
            ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect")

            ax.set_xlabel(f"Target {name}", fontsize=10)
            ax.set_ylabel(f"Predicted {name}", fontsize=10)
            ax.set_title(f"{name} Alignment", fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plot_path = self.results_dir / "objective_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Objective comparison saved to {plot_path}")

    def analyze_pareto_front(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        objective_names: List[str],
        save_plot: bool = True,
    ) -> Dict[str, Any]:
        """Analyze Pareto front coverage.

        Args:
            predictions: Predicted scores [B, num_objectives]
            targets: Target scores [B, num_objectives]
            objective_names: Names of objectives
            save_plot: Whether to save plot

        Returns:
            Dictionary with Pareto analysis
        """
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Compute Pareto front for targets
        target_pareto = self._compute_pareto_front(targets_np)
        pred_pareto = self._compute_pareto_front(predictions_np)

        analysis = {
            "target_pareto_size": len(target_pareto),
            "pred_pareto_size": len(pred_pareto),
            "pareto_coverage": float(len(pred_pareto) / len(predictions_np)),
        }

        if save_plot and len(objective_names) >= 2:
            self._plot_pareto_front(
                predictions_np, targets_np, pred_pareto, target_pareto, objective_names
            )

        return analysis

    def _compute_pareto_front(self, points: np.ndarray) -> List[int]:
        """Compute indices of points on the Pareto front.

        Args:
            points: Array of points [N, D]

        Returns:
            List of indices of Pareto-optimal points
        """
        n_points = len(points)
        is_pareto = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(n_points):
                if i != j and is_pareto[i]:
                    # Check if j dominates i (better in all dimensions)
                    if np.all(points[j] >= points[i]) and np.any(
                        points[j] > points[i]
                    ):
                        is_pareto[i] = False
                        break

        return np.where(is_pareto)[0].tolist()

    def _plot_pareto_front(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        pred_pareto: List[int],
        target_pareto: List[int],
        objective_names: List[str],
    ) -> None:
        """Plot Pareto front visualization.

        Args:
            predictions: Predicted scores
            targets: Target scores
            pred_pareto: Indices of predicted Pareto points
            target_pareto: Indices of target Pareto points
            objective_names: Names of objectives
        """
        plt.figure(figsize=(10, 8))

        # Plot predictions
        plt.scatter(
            predictions[:, 0],
            predictions[:, 1],
            alpha=0.3,
            s=30,
            c="blue",
            label="Predictions",
        )

        # Plot targets
        plt.scatter(
            targets[:, 0],
            targets[:, 1],
            alpha=0.3,
            s=30,
            c="green",
            label="Targets",
        )

        # Highlight Pareto fronts
        if pred_pareto:
            plt.scatter(
                predictions[pred_pareto, 0],
                predictions[pred_pareto, 1],
                s=100,
                c="red",
                marker="*",
                label="Predicted Pareto",
                edgecolors="black",
            )

        if target_pareto:
            plt.scatter(
                targets[target_pareto, 0],
                targets[target_pareto, 1],
                s=100,
                c="orange",
                marker="D",
                label="Target Pareto",
                edgecolors="black",
            )

        plt.xlabel(objective_names[0], fontsize=12)
        plt.ylabel(objective_names[1], fontsize=12)
        plt.title("Pareto Front Analysis", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plot_path = self.results_dir / "pareto_front.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Pareto front plot saved to {plot_path}")

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        output_path = self.results_dir / filename

        # Convert numpy/torch types to native Python types
        metrics_serializable = self._make_serializable(metrics)

        with open(output_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)

        logger.info(f"Metrics saved to {output_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return obj
