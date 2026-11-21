from __future__ import annotations

from typing import Callable

from ...common.errors import NotFound
from .base import ColumnInfo, Connector, DatasetRef, DatasetSchema, DeclaredKey

ConnectorFactory = Callable[..., Connector]  # (settings, secrets) -> Connector


def build_connector(settings, secrets=None) -> Connector:
    if settings.kind == "database":
        from .database import DatabaseConnector

        return DatabaseConnector(settings, secrets)
    if settings.kind == "parquet":
        from .parquet import ParquetConnector

        return ParquetConnector(settings, secrets)
    if settings.kind == "databricks":
        from .databricks import DatabricksConnector

        return DatabricksConnector(settings, secrets)
    raise NotFound(f"unknown connector kind {settings.kind}")


__all__ = ["build_connector", "Connector", "ConnectorFactory", "ColumnInfo", "DatasetRef", "DatasetSchema",
           "DeclaredKey"]
