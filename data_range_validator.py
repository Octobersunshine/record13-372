import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RangeRule:
    column: str
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    include_min: bool = True
    include_max: bool = True

    def validate(self, series: pd.Series) -> pd.Series:
        mask = pd.Series(True, index=series.index)

        if self.allowed_values is not None:
            mask = mask & series.isin(self.allowed_values)

        if self.min_value is not None:
            if self.include_min:
                mask = mask & (series >= self.min_value)
            else:
                mask = mask & (series > self.min_value)

        if self.max_value is not None:
            if self.include_max:
                mask = mask & (series <= self.max_value)
            else:
                mask = mask & (series < self.max_value)

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
