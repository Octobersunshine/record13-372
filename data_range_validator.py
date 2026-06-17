import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, date


def _is_date_like(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return True
    if isinstance(value, str):
        try:
            pd.to_datetime(value)
            return True
        except (ValueError, TypeError):
            return False
    return False


def _to_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
        return ts.timestamp()
    except (ValueError, TypeError):
        return None


def _series_to_timestamp(series: pd.Series) -> Optional[pd.Series]:
    try:
        dt_series = pd.to_datetime(series, errors="coerce")
        if dt_series.isna().all():
            return None
        return dt_series.apply(lambda x: x.timestamp() if pd.notna(x) else None)
    except (ValueError, TypeError):
        return None


@dataclass
class RangeRule:
    column: str
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    include_min: bool = True
    include_max: bool = True

    def _is_date_comparison(self, series: pd.Series) -> bool:
        has_date_boundary = _is_date_like(self.min_value) or _is_date_like(self.max_value)
        if has_date_boundary:
            return True
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        sample = series.dropna().head(5)
        if len(sample) > 0 and all(_is_date_like(v) for v in sample):
            return True
        return False

    def _prepare_compare_values(self, series: pd.Series):
        use_timestamp = self._is_date_comparison(series)

        if use_timestamp:
            series_ts = _series_to_timestamp(series)
            min_ts = _to_timestamp(self.min_value)
            max_ts = _to_timestamp(self.max_value)

            compare_series = series_ts if series_ts is not None else series
            compare_min = min_ts if min_ts is not None else self.min_value
            compare_max = max_ts if max_ts is not None else self.max_value
        else:
            compare_series = series
            compare_min = self.min_value
            compare_max = self.max_value

        return compare_series, compare_min, compare_max

    def validate_detail(self, series: pd.Series) -> Dict[str, pd.Series]:
        details = {
            "below_min": pd.Series(False, index=series.index),
            "above_max": pd.Series(False, index=series.index),
            "not_in_allowed": pd.Series(False, index=series.index),
        }

        compare_series, compare_min, compare_max = self._prepare_compare_values(series)

        if self.allowed_values is not None:
            details["not_in_allowed"] = ~series.isin(self.allowed_values)

        if compare_min is not None:
            if self.include_min:
                details["below_min"] = compare_series < compare_min
            else:
                details["below_min"] = compare_series <= compare_min

        if compare_max is not None:
            if self.include_max:
                details["above_max"] = compare_series > compare_max
            else:
                details["above_max"] = compare_series >= compare_max

        return details

    def validate(self, series: pd.Series) -> pd.Series:
        details = self.validate_detail(series)
        combined = pd.Series(False, index=series.index)
        for mask in details.values():
            combined = combined | mask
        return combined


@dataclass
class ValidationResult:
    is_valid: bool
    total_rows: int
    invalid_rows_count: int
    invalid_rows: pd.DataFrame
    violations: Dict[str, pd.Series] = field(default_factory=dict)
    violation_details: Dict[str, Dict[str, pd.Series]] = field(default_factory=dict)
    rules: List[RangeRule] = field(default_factory=list)

    def _get_rule_by_column(self, column: str) -> Optional[RangeRule]:
        for rule in self.rules:
            if rule.column == column:
                return rule
        return None

    def violation_statistics(self) -> pd.DataFrame:
        rows = []
        for col, total_mask in self.violations.items():
            total = int(total_mask.sum())
            if total == 0:
                continue

            details = self.violation_details.get(col, {})
            below_min = int(details.get("below_min", pd.Series(dtype=bool)).sum())
            above_max = int(details.get("above_max", pd.Series(dtype=bool)).sum())
            not_in_allowed = int(details.get("not_in_allowed", pd.Series(dtype=bool)).sum())

            rule = self._get_rule_by_column(col)
            min_val = rule.min_value if rule else None
            max_val = rule.max_value if rule else None

            rows.append({
                "column": col,
                "total_violations": total,
                "violation_rate": f"{total / self.total_rows * 100:.2f}%",
                "below_min": below_min,
                "above_max": above_max,
                "not_in_allowed": not_in_allowed,
                "min_value": str(min_val) if min_val is not None else "-",
                "max_value": str(max_val) if max_val is not None else "-",
            })

        if not rows:
            return pd.DataFrame(columns=[
                "column", "total_violations", "violation_rate",
                "below_min", "above_max", "not_in_allowed",
                "min_value", "max_value"
            ])

        df = pd.DataFrame(rows)
        df = df.sort_values("total_violations", ascending=False).reset_index(drop=True)
        return df

    def summary(self) -> Dict[str, Any]:
        stats = self.violation_statistics()
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "invalid_rows_count": self.invalid_rows_count,
            "invalid_rate": f"{self.invalid_rows_count / self.total_rows * 100:.2f}%" if self.total_rows > 0 else "0.00%",
            "violations_by_column": {
                col: int(viol.sum()) for col, viol in self.violations.items()
            },
            "violation_details": stats.to_dict(orient="records") if not stats.empty else [],
        }

    def statistics_report(self) -> str:
        stats_df = self.violation_statistics()

        lines = [
            "=" * 70,
            "范围违规统计报告",
            "=" * 70,
            f"总行数: {self.total_rows}",
            f"异常行数: {self.invalid_rows_count}",
            f"异常率: {self.invalid_rows_count / self.total_rows * 100:.2f}%" if self.total_rows > 0 else "异常率: 0.00%",
            f"涉及字段数: {len(stats_df)}",
            "",
        ]

        if stats_df.empty:
            lines.append("✓ 所有数据均在正常范围内")
        else:
            lines.append("各字段违规详情 (按违规数降序):")
            lines.append("-" * 70)

            col_width = max(len("字段"), max(len(c) for c in stats_df["column"])) + 2
            lines.append(
                f"{'字段'.ljust(col_width)}"
                f"{'违规数'.rjust(8)}"
                f"{'违规率'.rjust(10)}"
                f"{'低于下限'.rjust(10)}"
                f"{'超出上限'.rjust(10)}"
                f"{'不在允许值'.rjust(12)}"
            )
            lines.append("-" * 70)

            for _, row in stats_df.iterrows():
                lines.append(
                    f"{str(row['column']).ljust(col_width)}"
                    f"{str(row['total_violations']).rjust(8)}"
                    f"{str(row['violation_rate']).rjust(10)}"
                    f"{str(row['below_min']).rjust(10)}"
                    f"{str(row['above_max']).rjust(10)}"
                    f"{str(row['not_in_allowed']).rjust(12)}"
                )

            lines.append("-" * 70)
            lines.append("")
            lines.append("各字段范围定义:")
            for _, row in stats_df.iterrows():
                lines.append(f"  {row['column']}: [{row['min_value']}, {row['max_value']}]")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def detailed_report(self) -> str:
        lines = [
            "=" * 60,
            "数据范围校验报告",
            "=" * 60,
            f"总行数: {self.total_rows}",
            f"异常行数: {self.invalid_rows_count}",
            f"校验结果: {'通过' if self.is_valid else '未通过'}",
            "",
        ]

        if self.violations:
            lines.append("各列违规统计:")
            for col, viol in self.violations.items():
                count = int(viol.sum())
                if count > 0:
                    pct = count / self.total_rows * 100 if self.total_rows > 0 else 0
                    lines.append(f"  - {col}: {count} 行超出范围 ({pct:.2f}%)")

        if not self.invalid_rows.empty:
            lines.append("")
            lines.append("异常数据详情:")
            lines.append("-" * 60)
            lines.append(self.invalid_rows.to_string())

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


class DataRangeValidator:
    def __init__(self, rules: Optional[List[RangeRule]] = None):
        self.rules: List[RangeRule] = rules or []

    def add_rule(
        self,
        column: str,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        allowed_values: Optional[List[Any]] = None,
        include_min: bool = True,
        include_max: bool = True,
    ) -> "DataRangeValidator":
        self.rules.append(
            RangeRule(
                column=column,
                min_value=min_value,
                max_value=max_value,
                allowed_values=allowed_values,
                include_min=include_min,
                include_max=include_max,
            )
        )
        return self

    def add_rules_from_dict(self, rules_dict: Dict[str, Dict[str, Any]]) -> "DataRangeValidator":
        for column, config in rules_dict.items():
            self.add_rule(column=column, **config)
        return self

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        violations: Dict[str, pd.Series] = {}
        violation_details: Dict[str, Dict[str, pd.Series]] = {}
        combined_mask = pd.Series(False, index=df.index)

        for rule in self.rules:
            if rule.column not in df.columns:
                raise ValueError(f"列 '{rule.column}' 不存在于数据中")

            detail_masks = rule.validate_detail(df[rule.column])
            violation_details[rule.column] = detail_masks

            viol_mask = pd.Series(False, index=df.index)
            for mask in detail_masks.values():
                viol_mask = viol_mask | mask
            violations[rule.column] = viol_mask

            combined_mask = combined_mask | viol_mask

        invalid_rows = df[combined_mask].copy()
        for col, mask in violations.items():
            invalid_rows[f"{col}_out_of_range"] = mask[combined_mask]

        return ValidationResult(
            is_valid=not combined_mask.any(),
            total_rows=len(df),
            invalid_rows_count=int(combined_mask.sum()),
            invalid_rows=invalid_rows,
            violations=violations,
            violation_details=violation_details,
            rules=self.rules,
        )

    def filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.validate(df)
        valid_mask = ~pd.Series(False, index=df.index)
        for mask in result.violations.values():
            valid_mask = valid_mask & ~mask
        return df[valid_mask].copy()
