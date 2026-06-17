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

    def validate(self, series: pd.Series) -> pd.Series:
        mask = pd.Series(True, index=series.index)

        if self.allowed_values is not None:
            mask = mask & series.isin(self.allowed_values)

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

        if compare_min is not None:
            if self.include_min:
                mask = mask & (compare_series >= compare_min)
            else:
                mask = mask & (compare_series > compare_min)

        if compare_max is not None:
            if self.include_max:
                mask = mask & (compare_series <= compare_max)
            else:
                mask = mask & (compare_series < compare_max)

        return ~mask


@dataclass
class ValidationResult:
    is_valid: bool
    total_rows: int
    invalid_rows_count: int
    invalid_rows: pd.DataFrame
    violations: Dict[str, pd.Series] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "invalid_rows_count": self.invalid_rows_count,
            "violations_by_column": {
                col: int(viol.sum()) for col, viol in self.violations.items()
            },
        }

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
                    lines.append(f"  - {col}: {count} 行超出范围")

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
        combined_mask = pd.Series(False, index=df.index)

        for rule in self.rules:
            if rule.column not in df.columns:
                raise ValueError(f"列 '{rule.column}' 不存在于数据中")

            viol_mask = rule.validate(df[rule.column])
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
        )

    def filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.validate(df)
        valid_mask = ~pd.Series(False, index=df.index)
        for mask in result.violations.values():
            valid_mask = valid_mask & ~mask
        return df[valid_mask].copy()
