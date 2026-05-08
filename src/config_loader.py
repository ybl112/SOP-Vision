"""
YAML 全局配置加载器。

用法:
    from src.config_loader import load_config
    cfg = load_config("config/config.yaml")
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径，默认查找项目根目录下的 config/config.yaml

    Returns:
        dict: 配置字典
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "config.yaml",
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg
