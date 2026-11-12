# -*- coding: utf-8 -*-
"""
Vector store stage: embeds the pushed resource into Pinecone.

Port of the post-push embedding block from the ``vector_storage_v2`` branch
(the monolithic ``jobs.py``) onto the staged pipeline. Runs after the
datastore load and metadata stages. Non-blocking by design: any failure is
logged and the job still completes, as on the original branch.
"""

import os
import time
from typing import Any, Dict, Optional

from dateutil.parser import parse as parsedate

import ckanext.datapusher_plus.config as conf
import ckanext.datapusher_plus.datastore_utils as dsu
from ckanext.datapusher_plus.jobs.context import ProcessingContext
from ckanext.datapusher_plus.jobs.stages.base import BaseStage

DATE_TYPES = ("Date", "DateTime")


def temporal_coverage(stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Year range covered by the data, from qsv stats min/max.

    ``stats`` is ``context.resource_fields_stats``: ``{field: {"stats": row}}``.
    Year-named columns win; otherwise Date/DateTime columns are used.
    """
    year_cols, date_cols = {}, {}
    for field, entry in stats.items():
        row = entry.get("stats") or {}
        if "year" in field.lower():
            year_cols[field] = row
        elif row.get("type") in DATE_TYPES:
            date_cols[field] = row

    years = set()
    for row in year_cols.values():
        try:
            years.update(range(int(float(row["min"])), int(float(row["max"])) + 1))
        except (KeyError, ValueError, TypeError):
            pass
    if not years:
        for row in date_cols.values():
            try:
                years.update(
                    range(parsedate(str(row["min"])).year, parsedate(str(row["max"])).year + 1)
                )
            except (KeyError, ValueError, TypeError, OverflowError):
                pass
    if not years:
        return None
    return {
        "min": min(years),
        "max": max(years),
        "all_years": sorted(years),
        "year_count": len(years),
        "temporal_columns": {col: True for col in {**year_cols, **date_cols}},
    }


class VectorStoreStage(BaseStage):
    """Embed the resource profile (stats, freqs, AI description) in Pinecone."""

    def should_skip(self, context: ProcessingContext) -> bool:
        return not conf.ENABLE_VECTOR_STORE

    def process(self, context: ProcessingContext) -> ProcessingContext:
        start = time.perf_counter()
        try:
            from ckanext.datapusher_plus.vector_store import DataPusherVectorStore

            vector_store = DataPusherVectorStore()
            if not vector_store.enabled:
                return context

            sample_data = None
            if conf.VECTOR_INCLUDE_SAMPLE_DATA:
                try:
                    sample_csv = os.path.join(context.temp_dir, "qsv_sample_for_llm.csv")
                    context.qsv.slice(
                        context.tmp, length=conf.VECTOR_LLM_SAMPLE_ROWS, output_file=sample_csv
                    )
                    with open(sample_csv, "r", encoding="utf-8") as f:
                        sample_data = f.read()
                except Exception as e:
                    context.logger.warning(f"Could not generate sample data: {e}")

            temporal_info = None
            if conf.DETECT_TEMPORAL_COVERAGE:
                temporal_info = temporal_coverage(context.resource_fields_stats)
                if temporal_info:
                    context.logger.info(
                        f"Detected temporal coverage: {temporal_info['min']} to {temporal_info['max']}"
                    )

            resource = dict(context.resource)
            resource["record_count"] = context.copied_count or context.rows_to_copy

            ok = vector_store.embed_resource(
                resource_id=context.resource_id,
                resource_metadata=resource,
                dataset_metadata=dsu.get_package(context.resource["package_id"]),
                stats_data=context.resource_fields_stats,
                freq_data=context.resource_fields_freqs,
                temporal_info=temporal_info,
                sample_data=sample_data,
                logger=context.logger,
            )
            if ok:
                context.logger.info(f"Resource {context.resource_id} embedded in vector store")
            else:
                context.logger.warning("Vector store embedding failed")
        except Exception as e:
            # Never fail the job over embedding (same contract as AISuggestionsStage).
            context.logger.error(f"Error during vector store embedding: {e}")

        context.logger.info(
            f"Vector store embedding took {time.perf_counter() - start:,.2f} seconds"
        )
        return context


if __name__ == "__main__":
    s = {
        "fiscal_year": {"stats": {"type": "Integer", "min": "2019", "max": "2021"}},
        "updated": {"stats": {"type": "Date", "min": "2001-01-05", "max": "2002-03-01"}},
        "name": {"stats": {"type": "String", "min": "a", "max": "z"}},
    }
    t = temporal_coverage(s)
    assert (t["min"], t["max"], t["year_count"]) == (2019, 2021, 3), t
    assert set(t["temporal_columns"]) == {"fiscal_year", "updated"}
    del s["fiscal_year"]
    assert temporal_coverage(s)["all_years"] == [2001, 2002]
    assert temporal_coverage({"name": s["name"]}) is None
    print("temporal_coverage ok")
