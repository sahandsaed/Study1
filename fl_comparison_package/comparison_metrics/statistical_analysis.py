"""
Statistical Analysis Module for FL Framework Comparison
Provides statistical significance tests and effect size calculations

This module performs rigorous statistical analysis to validate
the comparison results between TFF and Flower frameworks.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_interpretation: str
    is_significant: bool
    confidence_level: float = 0.95
    conclusion: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a sample."""
    name: str
    n: int
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ComparisonAnalysis:
    """Complete comparison analysis results."""
    metric_name: str
    tff_stats: DescriptiveStats
    flower_stats: DescriptiveStats
    normality_tff: StatisticalTestResult
    normality_flower: StatisticalTestResult
    comparison_test: StatisticalTestResult
    
    def to_dict(self) -> Dict:
        return {
            'metric_name': self.metric_name,
            'tff_stats': self.tff_stats.to_dict(),
            'flower_stats': self.flower_stats.to_dict(),
            'normality_tff': self.normality_tff.to_dict(),
            'normality_flower': self.normality_flower.to_dict(),
            'comparison_test': self.comparison_test.to_dict()
        }


class StatisticalAnalyzer:
    """
    Performs statistical analysis for FL framework comparison.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
    
    def calculate_descriptive_stats(
        self,
        data: np.ndarray,
        name: str
    ) -> DescriptiveStats:
        """Calculate descriptive statistics for a dataset."""
        data = np.array(data)
        
        return DescriptiveStats(
            name=name,
            n=len(data),
            mean=float(np.mean(data)),
            std=float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
            median=float(np.median(data)),
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
            q1=float(np.percentile(data, 25)),
            q3=float(np.percentile(data, 75)),
            iqr=float(np.percentile(data, 75) - np.percentile(data, 25)),
            skewness=float(stats.skew(data)) if len(data) > 2 else 0.0,
            kurtosis=float(stats.kurtosis(data)) if len(data) > 3 else 0.0
        )
    
    def test_normality(
        self,
        data: np.ndarray,
        name: str
    ) -> StatisticalTestResult:
        """
        Test if data follows normal distribution using Shapiro-Wilk test.
        
        For small samples (n < 50), Shapiro-Wilk is most appropriate.
        """
        data = np.array(data)
        
        if len(data) < 3:
            return StatisticalTestResult(
                test_name="Shapiro-Wilk",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_interpretation="N/A (insufficient data)",
                is_significant=False,
                confidence_level=self.confidence_level,
                conclusion=f"Insufficient data for normality test (n={len(data)})"
            )
        
        statistic, p_value = stats.shapiro(data)
        is_normal = p_value > self.alpha
        
        return StatisticalTestResult(
            test_name="Shapiro-Wilk",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=0.0,  # N/A for normality tests
            effect_size_interpretation="N/A",
            is_significant=not is_normal,
            confidence_level=self.confidence_level,
            conclusion=f"{name} {'follows' if is_normal else 'does not follow'} normal distribution (p={p_value:.4f})"
        )
    
    def cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Tuple[float, str]:
        """
        Calculate Cohen's d effect size.
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0, "negligible"
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return float(d), interpretation
    
    def rank_biserial_correlation(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        U_statistic: float
    ) -> Tuple[float, str]:
        """
        Calculate rank-biserial correlation as effect size for Mann-Whitney U.
        
        r = 1 - (2U)/(n1*n2)
        
        Interpretation similar to Cohen's d.
        """
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * U_statistic) / (n1 * n2)
        
        abs_r = abs(r)
        if abs_r < 0.1:
            interpretation = "negligible"
        elif abs_r < 0.3:
            interpretation = "small"
        elif abs_r < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return float(r), interpretation
    
    def independent_t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        name1: str = "Group 1",
        name2: str = "Group 2",
        equal_var: bool = True
    ) -> StatisticalTestResult:
        """
        Perform independent samples t-test.
        
        Use when both groups are normally distributed.
        """
        group1, group2 = np.array(group1), np.array(group2)
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # Effect size
        effect_size, interpretation = self.cohens_d(group1, group2)
        
        is_significant = p_value < self.alpha
        
        # Determine which is better
        mean_diff = np.mean(group1) - np.mean(group2)
        better = name1 if mean_diff < 0 else name2  # Assuming lower is better
        
        conclusion = f"{'Significant' if is_significant else 'No significant'} difference found. "
        if is_significant:
            conclusion += f"{better} performs better (p={p_value:.4f}, d={effect_size:.3f} [{interpretation}])"
        else:
            conclusion += f"The frameworks perform similarly (p={p_value:.4f})"
        
        return StatisticalTestResult(
            test_name="Independent t-test" + (" (Welch's)" if not equal_var else ""),
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=effect_size,
            effect_size_interpretation=interpretation,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            conclusion=conclusion
        )
    
    def mann_whitney_u_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        name1: str = "Group 1",
        name2: str = "Group 2"
    ) -> StatisticalTestResult:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Use when normality assumption is violated.
        """
        group1, group2 = np.array(group1), np.array(group2)
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Effect size
        effect_size, interpretation = self.rank_biserial_correlation(
            group1, group2, statistic
        )
        
        is_significant = p_value < self.alpha
        
        # Determine which is better based on medians
        median_diff = np.median(group1) - np.median(group2)
        better = name1 if median_diff < 0 else name2
        
        conclusion = f"{'Significant' if is_significant else 'No significant'} difference found. "
        if is_significant:
            conclusion += f"{better} performs better (p={p_value:.4f}, r={effect_size:.3f} [{interpretation}])"
        else:
            conclusion += f"The frameworks perform similarly (p={p_value:.4f})"
        
        return StatisticalTestResult(
            test_name="Mann-Whitney U test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=effect_size,
            effect_size_interpretation=interpretation,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            conclusion=conclusion
        )
    
    def paired_t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        name1: str = "Group 1",
        name2: str = "Group 2"
    ) -> StatisticalTestResult:
        """
        Perform paired samples t-test.
        
        Use when samples are paired/matched.
        """
        group1, group2 = np.array(group1), np.array(group2)
        
        if len(group1) != len(group2):
            raise ValueError("Paired t-test requires equal sample sizes")
        
        statistic, p_value = stats.ttest_rel(group1, group2)
        
        # Effect size for paired samples (Cohen's dz)
        diff = group1 - group2
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
        
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        is_significant = p_value < self.alpha
        
        conclusion = f"{'Significant' if is_significant else 'No significant'} difference found (p={p_value:.4f}, d={effect_size:.3f} [{interpretation}])"
        
        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_interpretation=interpretation,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            conclusion=conclusion
        )
    
    def wilcoxon_signed_rank_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        name1: str = "Group 1",
        name2: str = "Group 2"
    ) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric paired test).
        
        Use when paired data doesn't meet normality assumption.
        """
        group1, group2 = np.array(group1), np.array(group2)
        
        if len(group1) != len(group2):
            raise ValueError("Wilcoxon test requires equal sample sizes")
        
        try:
            statistic, p_value = stats.wilcoxon(group1, group2)
        except ValueError:
            # All differences are zero
            return StatisticalTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_interpretation="negligible",
                is_significant=False,
                confidence_level=self.confidence_level,
                conclusion="No difference between groups"
            )
        
        # Effect size (r = Z / sqrt(N))
        n = len(group1)
        z = stats.norm.ppf(1 - p_value / 2)  # Approximate Z
        effect_size = z / np.sqrt(n) if n > 0 else 0.0
        
        abs_r = abs(effect_size)
        if abs_r < 0.1:
            interpretation = "negligible"
        elif abs_r < 0.3:
            interpretation = "small"
        elif abs_r < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        is_significant = p_value < self.alpha
        
        conclusion = f"{'Significant' if is_significant else 'No significant'} difference found (p={p_value:.4f}, r={effect_size:.3f} [{interpretation}])"
        
        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_interpretation=interpretation,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            conclusion=conclusion
        )
    
    def compare_metrics(
        self,
        tff_data: np.ndarray,
        flower_data: np.ndarray,
        metric_name: str
    ) -> ComparisonAnalysis:
        """
        Perform complete comparison analysis for a metric.
        
        Automatically selects appropriate test based on data characteristics.
        """
        tff_data = np.array(tff_data)
        flower_data = np.array(flower_data)
        
        # Descriptive statistics
        tff_stats = self.calculate_descriptive_stats(tff_data, "TFF")
        flower_stats = self.calculate_descriptive_stats(flower_data, "Flower")
        
        # Normality tests
        normality_tff = self.test_normality(tff_data, "TFF")
        normality_flower = self.test_normality(flower_data, "Flower")
        
        # Choose appropriate test
        both_normal = not normality_tff.is_significant and not normality_flower.is_significant
        
        if both_normal and len(tff_data) >= 5 and len(flower_data) >= 5:
            # Use parametric test (t-test)
            # Check for equal variances using Levene's test
            _, levene_p = stats.levene(tff_data, flower_data)
            equal_var = levene_p > self.alpha
            comparison_test = self.independent_t_test(
                tff_data, flower_data, "TFF", "Flower", equal_var
            )
        else:
            # Use non-parametric test (Mann-Whitney U)
            comparison_test = self.mann_whitney_u_test(
                tff_data, flower_data, "TFF", "Flower"
            )
        
        return ComparisonAnalysis(
            metric_name=metric_name,
            tff_stats=tff_stats,
            flower_stats=flower_stats,
            normality_tff=normality_tff,
            normality_flower=normality_flower,
            comparison_test=comparison_test
        )


class SurveyAnalyzer:
    """
    Analyzes survey data comparing FL framework preferences.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.statistical_analyzer = StatisticalAnalyzer(alpha)
    
    def analyze_likert_responses(
        self,
        tff_ratings: List[int],
        flower_ratings: List[int],
        aspect: str
    ) -> Dict[str, Any]:
        """
        Analyze Likert scale responses for a specific aspect.
        
        Args:
            tff_ratings: List of ratings for TFF (1-5 scale)
            flower_ratings: List of ratings for Flower (1-5 scale)
            aspect: Name of the aspect being rated
        
        Returns:
            Analysis results including statistics and tests
        """
        tff = np.array(tff_ratings)
        flower = np.array(flower_ratings)
        
        # Descriptive stats
        tff_stats = self.statistical_analyzer.calculate_descriptive_stats(tff, "TFF")
        flower_stats = self.statistical_analyzer.calculate_descriptive_stats(flower, "Flower")
        
        # Mann-Whitney U test (appropriate for ordinal data)
        comparison = self.statistical_analyzer.mann_whitney_u_test(
            tff, flower, "TFF", "Flower"
        )
        
        return {
            'aspect': aspect,
            'tff_stats': tff_stats.to_dict(),
            'flower_stats': flower_stats.to_dict(),
            'comparison': comparison.to_dict()
        }
    
    def analyze_preference_distribution(
        self,
        preferences: List[str]  # List of "TFF" or "Flower"
    ) -> Dict[str, Any]:
        """
        Analyze overall preference distribution using chi-square test.
        """
        tff_count = preferences.count("TFF")
        flower_count = preferences.count("Flower")
        total = len(preferences)
        
        # Expected frequencies (equal preference)
        expected = total / 2
        
        # Chi-square goodness of fit test
        chi2, p_value = stats.chisquare([tff_count, flower_count])
        
        is_significant = p_value < self.alpha
        
        return {
            'tff_count': tff_count,
            'flower_count': flower_count,
            'tff_percentage': tff_count / total * 100 if total > 0 else 0,
            'flower_percentage': flower_count / total * 100 if total > 0 else 0,
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'conclusion': f"{'Significant' if is_significant else 'No significant'} preference difference (p={p_value:.4f})"
        }
    
    def analyze_familiarity_effect(
        self,
        familiar_preferences: List[str],
        unfamiliar_preferences: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze if FL familiarity affects framework preference.
        Uses chi-square test for independence.
        """
        # Create contingency table
        familiar_tff = familiar_preferences.count("TFF")
        familiar_flower = familiar_preferences.count("Flower")
        unfamiliar_tff = unfamiliar_preferences.count("TFF")
        unfamiliar_flower = unfamiliar_preferences.count("Flower")
        
        contingency_table = np.array([
            [familiar_tff, familiar_flower],
            [unfamiliar_tff, unfamiliar_flower]
        ])
        
        # Chi-square test for independence
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Effect size (Cramér's V)
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0
        
        is_significant = p_value < self.alpha
        
        return {
            'contingency_table': contingency_table.tolist(),
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'cramers_v': float(cramers_v),
            'is_significant': is_significant,
            'conclusion': f"FL familiarity {'does' if is_significant else 'does not'} significantly affect framework preference (p={p_value:.4f})"
        }


def generate_synthetic_experiment_data(
    num_runs: int = 10,
    num_rounds: int = 10
) -> Dict[str, Dict[str, List[float]]]:
    """
    Generate synthetic experiment data for demonstration.
    
    In real usage, this would be replaced with actual experiment results.
    """
    np.random.seed(42)
    
    data = {
        'accuracy': {
            'tff': list(np.random.normal(0.85, 0.03, num_runs)),
            'flower': list(np.random.normal(0.86, 0.025, num_runs))
        },
        'training_time_seconds': {
            'tff': list(np.random.normal(120, 15, num_runs)),
            'flower': list(np.random.normal(100, 12, num_runs))
        },
        'communication_mb': {
            'tff': list(np.random.normal(50, 5, num_runs)),
            'flower': list(np.random.normal(45, 4, num_runs))
        },
        'memory_usage_mb': {
            'tff': list(np.random.normal(2000, 200, num_runs)),
            'flower': list(np.random.normal(1800, 180, num_runs))
        },
        'round_latency_ms': {
            'tff': list(np.random.normal(150, 20, num_runs * num_rounds)),
            'flower': list(np.random.normal(130, 18, num_runs * num_rounds))
        }
    }
    
    return data


def generate_synthetic_survey_data(
    num_familiar: int = 15,
    num_unfamiliar: int = 15
) -> Dict[str, Any]:
    """
    Generate synthetic survey data for demonstration.
    
    Survey aspects (1-5 Likert scale):
    - Ease of installation
    - Documentation quality
    - API usability
    - Debugging ease
    - Overall satisfaction
    """
    np.random.seed(42)
    
    # Familiar participants (tend to rate based on technical merits)
    familiar_data = {
        'ease_of_installation': {
            'tff': list(np.random.choice([2, 3, 3, 4], num_familiar)),
            'flower': list(np.random.choice([4, 4, 5, 5], num_familiar))
        },
        'documentation_quality': {
            'tff': list(np.random.choice([3, 4, 4, 4, 5], num_familiar)),
            'flower': list(np.random.choice([3, 3, 4, 4, 4], num_familiar))
        },
        'api_usability': {
            'tff': list(np.random.choice([2, 3, 3, 4], num_familiar)),
            'flower': list(np.random.choice([4, 4, 4, 5], num_familiar))
        },
        'debugging_ease': {
            'tff': list(np.random.choice([2, 2, 3, 3], num_familiar)),
            'flower': list(np.random.choice([3, 4, 4, 5], num_familiar))
        },
        'overall_satisfaction': {
            'tff': list(np.random.choice([3, 3, 4, 4], num_familiar)),
            'flower': list(np.random.choice([4, 4, 4, 5], num_familiar))
        },
        'preference': list(np.random.choice(['TFF', 'Flower', 'Flower', 'Flower'], num_familiar))
    }
    
    # Unfamiliar participants (tend to prefer simpler setup)
    unfamiliar_data = {
        'ease_of_installation': {
            'tff': list(np.random.choice([2, 2, 3, 3], num_unfamiliar)),
            'flower': list(np.random.choice([4, 4, 5, 5], num_unfamiliar))
        },
        'documentation_quality': {
            'tff': list(np.random.choice([3, 3, 4, 4], num_unfamiliar)),
            'flower': list(np.random.choice([3, 4, 4, 4], num_unfamiliar))
        },
        'api_usability': {
            'tff': list(np.random.choice([2, 2, 3, 3], num_unfamiliar)),
            'flower': list(np.random.choice([4, 4, 5, 5], num_unfamiliar))
        },
        'debugging_ease': {
            'tff': list(np.random.choice([2, 2, 2, 3], num_unfamiliar)),
            'flower': list(np.random.choice([3, 4, 4, 4], num_unfamiliar))
        },
        'overall_satisfaction': {
            'tff': list(np.random.choice([2, 3, 3, 3], num_unfamiliar)),
            'flower': list(np.random.choice([4, 4, 5, 5], num_unfamiliar))
        },
        'preference': list(np.random.choice(['TFF', 'Flower', 'Flower', 'Flower', 'Flower'], num_unfamiliar))
    }
    
    return {
        'familiar': familiar_data,
        'unfamiliar': unfamiliar_data
    }


def run_statistical_analysis(
    experiment_data: Optional[Dict] = None,
    survey_data: Optional[Dict] = None,
    output_dir: str = './results'
) -> Dict[str, Any]:
    """
    Run complete statistical analysis.
    
    Args:
        experiment_data: Experiment results (or None to use synthetic data)
        survey_data: Survey results (or None to use synthetic data)
        output_dir: Directory to save results
    
    Returns:
        Complete analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Use synthetic data if not provided
    if experiment_data is None:
        experiment_data = generate_synthetic_experiment_data()
    if survey_data is None:
        survey_data = generate_synthetic_survey_data()
    
    analyzer = StatisticalAnalyzer()
    survey_analyzer = SurveyAnalyzer()
    
    results = {
        'experiment_analysis': {},
        'survey_analysis': {}
    }
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS REPORT")
    print("TensorFlow Federated vs Flower Framework Comparison")
    print("="*70)
    
    # Analyze experiment metrics
    print("\n" + "-"*50)
    print("1. EXPERIMENT METRICS ANALYSIS")
    print("-"*50)
    
    for metric_name, metric_data in experiment_data.items():
        print(f"\n--- {metric_name.upper().replace('_', ' ')} ---")
        
        analysis = analyzer.compare_metrics(
            np.array(metric_data['tff']),
            np.array(metric_data['flower']),
            metric_name
        )
        
        print(f"TFF: mean={analysis.tff_stats.mean:.4f}, std={analysis.tff_stats.std:.4f}")
        print(f"Flower: mean={analysis.flower_stats.mean:.4f}, std={analysis.flower_stats.std:.4f}")
        print(f"Test: {analysis.comparison_test.test_name}")
        print(f"Result: {analysis.comparison_test.conclusion}")
        
        results['experiment_analysis'][metric_name] = analysis.to_dict()
    
    # Analyze survey data
    print("\n" + "-"*50)
    print("2. SURVEY ANALYSIS")
    print("-"*50)
    
    # Combine familiar and unfamiliar data for overall analysis
    all_familiar = survey_data['familiar']
    all_unfamiliar = survey_data['unfamiliar']
    
    aspects = ['ease_of_installation', 'documentation_quality', 'api_usability', 
               'debugging_ease', 'overall_satisfaction']
    
    print("\n--- LIKERT SCALE ANALYSIS ---")
    for aspect in aspects:
        # Combine data from both groups
        tff_ratings = all_familiar[aspect]['tff'] + all_unfamiliar[aspect]['tff']
        flower_ratings = all_familiar[aspect]['flower'] + all_unfamiliar[aspect]['flower']
        
        aspect_analysis = survey_analyzer.analyze_likert_responses(
            tff_ratings, flower_ratings, aspect
        )
        
        print(f"\n{aspect.upper().replace('_', ' ')}:")
        print(f"  TFF: mean={aspect_analysis['tff_stats']['mean']:.2f}")
        print(f"  Flower: mean={aspect_analysis['flower_stats']['mean']:.2f}")
        print(f"  {aspect_analysis['comparison']['conclusion']}")
        
        results['survey_analysis'][aspect] = aspect_analysis
    
    # Preference analysis
    print("\n--- PREFERENCE ANALYSIS ---")
    all_preferences = all_familiar['preference'] + all_unfamiliar['preference']
    preference_analysis = survey_analyzer.analyze_preference_distribution(all_preferences)
    
    print(f"TFF preferred: {preference_analysis['tff_count']} ({preference_analysis['tff_percentage']:.1f}%)")
    print(f"Flower preferred: {preference_analysis['flower_count']} ({preference_analysis['flower_percentage']:.1f}%)")
    print(f"{preference_analysis['conclusion']}")
    
    results['survey_analysis']['preference_distribution'] = preference_analysis
    
    # Familiarity effect
    print("\n--- FAMILIARITY EFFECT ---")
    familiarity_analysis = survey_analyzer.analyze_familiarity_effect(
        all_familiar['preference'],
        all_unfamiliar['preference']
    )
    print(f"{familiarity_analysis['conclusion']}")
    
    results['survey_analysis']['familiarity_effect'] = familiarity_analysis
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    significant_metrics = []
    for metric, analysis in results['experiment_analysis'].items():
        if analysis['comparison_test']['is_significant']:
            significant_metrics.append(metric)
    
    print(f"\nSignificant differences found in {len(significant_metrics)} metrics:")
    for metric in significant_metrics:
        print(f"  - {metric}")
    
    if preference_analysis['is_significant']:
        preferred = "Flower" if preference_analysis['flower_count'] > preference_analysis['tff_count'] else "TFF"
        print(f"\nOverall framework preference: {preferred}")
    else:
        print("\nNo significant overall preference between frameworks")
    
    # Save results
    results_path = os.path.join(output_dir, 'statistical_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    results = run_statistical_analysis(output_dir='./results')
