import pandas as pd
from data_range_validator import DataRangeValidator, RangeRule
from datetime import datetime


def create_sample_data() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "age": [25, 17, 30, 120, 45, -5, 28, 35],
        "score": [85.5, 92.0, 105.0, 67.3, -1.0, 78.0, 88.5, 95.0],
        "grade": ["A", "B", "C", "F", "A", "Z", "B", "A"],
        "salary": [50000, 80000, 1500000, 60000, 45000, 75000, 90000, 120000],
        "register_date": pd.to_datetime([
            "2023-01-15", "2022-06-20", "2025-03-10",
            "2023-08-05", "2020-12-01", "2024-04-18",
            "2023-11-22", "2024-02-29"
        ]),
    }
    return pd.DataFrame(data)


def example_basic_usage():
    print("=" * 60)
    print("示例1: 基础用法 - 链式添加规则")
    print("=" * 60)

    df = create_sample_data()
    print("原始数据:")
    print(df)
    print()

    validator = DataRangeValidator()
    validator.add_rule(column="age", min_value=18, max_value=100)
    validator.add_rule(column="score", min_value=0, max_value=100)
    validator.add_rule(column="grade", allowed_values=["A", "B", "C", "D", "F"])

    result = validator.validate(df)

    print(result.detailed_report())
    print()

    print("异常行（带标记列）:")
    print(result.invalid_rows)
    print()


def example_dict_config():
    print("=" * 60)
    print("示例2: 使用字典批量配置规则")
    print("=" * 60)

    df = create_sample_data()

    rules_dict = {
        "age": {"min_value": 18, "max_value": 100},
        "salary": {"min_value": 30000, "max_value": 1000000, "include_max": False},
        "register_date": {
            "min_value": pd.Timestamp("2021-01-01"),
            "max_value": pd.Timestamp("2024-12-31"),
        },
    }

    validator = DataRangeValidator()
    validator.add_rules_from_dict(rules_dict)

    result = validator.validate(df)

    print(result.detailed_report())
    print()

    print("校验结果摘要 (JSON格式):")
    summary = result.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()


def example_filter_valid():
    print("=" * 60)
    print("示例3: 过滤出有效数据")
    print("=" * 60)

    df = create_sample_data()

    validator = DataRangeValidator()
    validator.add_rule(column="age", min_value=18, max_value=100)
    validator.add_rule(column="score", min_value=0, max_value=100)

    valid_df = validator.filter_valid(df)

    print(f"原始数据行数: {len(df)}")
    print(f"有效数据行数: {len(valid_df)}")
    print()
    print("有效数据:")
    print(valid_df)
    print()


def example_with_rule_objects():
    print("=" * 60)
    print("示例4: 使用 RangeRule 对象直接初始化")
    print("=" * 60)

    df = create_sample_data()

    rules = [
        RangeRule(column="age", min_value=18, max_value=100),
        RangeRule(column="score", min_value=0, max_value=100, include_min=False),
        RangeRule(column="grade", allowed_values=["A", "B", "C"]),
    ]

    validator = DataRangeValidator(rules=rules)
    result = validator.validate(df)

    print(result.detailed_report())
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_dict_config()
    example_filter_valid()
    example_with_rule_objects()
