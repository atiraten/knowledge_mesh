"""Parquet files: ``path`` is a single file or a directory of ``*.parquet`` files (one dataset each)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ...common.errors import SourceUnavailable
from .base import ColumnInfo, Connector, DatasetRef, DatasetSchema

_SCHEMA = "files"


class ParquetConnector(Connector):
    def __init__(self, settings, secrets=None):
        super().__init__(settings, secrets)
        self.root = Path(settings.path)
        if not self.root.exists():
            raise SourceUnavailable(f"source {settings.id}: parquet path not found", {"path": str(self.root)})

    def _files(self) -> dict[str, Path]:
        if self.root.is_file():
            return {self.root.stem: self.root}
        files = {}
        for p in sorted(self.root.rglob("*.parquet")):
            rel = p.relative_to(self.root).with_suffix("")
            files[".".join(rel.parts)] = p
        return files

    def list_datasets(self) -> list[DatasetRef]:
        return [r for r in (DatasetRef(_SCHEMA, name, "file") for name in self._files()) if self.allowed(r)]

    def _path(self, ref: DatasetRef) -> Path:
        path = self._files().get(ref.name)
        if path is None:
            raise SourceUnavailable(f"dataset {ref.name} not found in source {self.source_id}")
        return path

    def describe(self, ref: DatasetRef) -> DatasetSchema:
        path = self._path(ref)
        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:
            raise SourceUnavailable(f"cannot read parquet file {path.name}: {type(exc).__name__}") from exc
        schema = pf.schema_arrow
        cols = [ColumnInfo(f.name, str(f.type), f.nullable, i) for i, f in enumerate(schema)]
        return DatasetSchema(ref, str(path), cols, [], pf.metadata.num_rows)

    def read_sample(self, ref: DatasetRef, columns: list[str], limit: int | None) -> pa.Table:
        pf = pq.ParquetFile(self._path(ref))
        if not limit:
            return pf.read(columns=columns)
        batches, remaining = [], limit
        for batch in pf.iter_batches(batch_size=min(65536, limit), columns=columns):
            batches.append(batch.slice(0, remaining))
            remaining -= min(remaining, batch.num_rows)
            if remaining <= 0:
                break
        if not batches:
            return pf.schema_arrow.empty_table().select(columns)
        return pa.Table.from_batches(batches)
