"""Command line entry point: ``knowledge-mesh <command>`` (or ``python src/harness.py <command>``)."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knowledge-mesh", description="Knowledge Mesh POC")
    parser.add_argument("--config", help="path to mesh.yaml (default: $KNOWLEDGE_MESH_CONFIG or config/mesh.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the FastAPI app and browser UI")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    mcp = sub.add_parser("mcp", help="run the MCP server")
    mcp.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    mcp.add_argument("--host", default="127.0.0.1")
    mcp.add_argument("--port", type=int, default=8765)

    sub.add_parser("check-config", help="validate config/mesh.yaml and print the resolved settings")

    ingest = sub.add_parser("ingest", help="ingest and index a file or directory under an authorized root")
    ingest.add_argument("path")

    crawl = sub.add_parser("crawl", help="crawl a website and index its pages")
    crawl.add_argument("url")
    crawl.add_argument("--max-depth", type=int, help="1 = this page only (no robots.txt, sitemap or links)")
    crawl.add_argument("--max-pages", type=int)

    catalog = sub.add_parser("catalog", help="catalog a structured source, discover keys and generate terms")
    catalog.add_argument("source_id")
    catalog.add_argument("--dataset", action="append", dest="datasets")

    search = sub.add_parser("search", help="search with the agent (plan, tool loop, cited answer)")
    search.add_argument("query")
    search.add_argument("--top-k", type=int)
    search.add_argument("--retrieve-only", action="store_true",
                        help="direct mixed retrieval without the agent")

    reindex = sub.add_parser("reindex", help="re-chunk and re-embed all documents (after chunking changes)")
    reindex.add_argument("--rebuild", action="store_true",
                         help="move both index directories aside and rebuild them (required after an embedding-model "
                              "change); old directories are kept as *.bak-<timestamp>")

    ev = sub.add_parser("eval", help="offline evaluation: retrieval (Recall@K, MRR, nDCG) or, with --agent, the "
                                     "full agent search (judge, compliance, recall, pass rate); optional Ragas")
    ev.add_argument("testset", help="JSON Lines test set")
    ev.add_argument("--k", type=int)
    ev.add_argument("--ragas", action="store_true")
    ev.add_argument("--agent", action="store_true", help="evaluate agent searches instead of retrieval")
    ev.add_argument("--min-pass-rate", type=float, default=0.8, help="--agent: exit 1 below this pass rate")
    ev.add_argument("--min-compliance", type=float, default=0.8)
    ev.add_argument("--min-faithfulness", type=float, default=0.7)

    args = parser.parse_args(argv)
    from .config import ConfigError, load_config

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "check-config":
        print(json.dumps(cfg.model_dump(mode="json", exclude={"base_dir"}), indent=2))
        return 0
    if args.command == "serve":
        import uvicorn

        from .api.app import create_app

        uvicorn.run(create_app(config_path=args.config), host=args.host or cfg.api.host, port=args.port or cfg.api.port,
                    log_config=None)
        return 0
    if args.command == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main(args.config, args.transport, args.host, args.port)
        return 0

    from .common.errors import MeshError
    from .mesh import KnowledgeMesh

    mesh = KnowledgeMesh(cfg)
    try:
        if args.command == "reindex" and args.rebuild:
            out = mesh.rebuild_indexes()
        elif args.command == "ingest":
            out = mesh.ingest_document(path=args.path, wait_seconds=3600)
        elif args.command == "crawl":
            out = mesh.crawl_website(args.url, args.max_depth, args.max_pages, wait_seconds=3600)
        elif args.command == "catalog":
            out = mesh.catalog_source(args.source_id, args.datasets, wait_seconds=3600)
        elif args.command == "search":
            out = (mesh.retrieve_documents([args.query], args.top_k) if args.retrieve_only
                   else mesh.search(args.query, args.top_k))
        elif args.command == "reindex":
            out = mesh.reindex_all_documents(wait_seconds=3600)
        elif args.command == "eval" and args.agent:
            from .evaluation.agent_eval import evaluate_agent, load_agent_testset

            out = evaluate_agent(mesh, load_agent_testset(args.testset), args.ragas, args.min_pass_rate,
                                 args.min_compliance, args.min_faithfulness)
            out = {k: v for k, v in out.items() if k != "cases"}
            print(json.dumps(out, indent=2, default=str))
            return 0 if out["summary"]["run_passed"] else 1
        elif args.command == "eval":
            from .evaluation.retrieval_eval import evaluate, load_testset

            out = evaluate(mesh, load_testset(args.testset), args.k, args.ragas)
            out = {k: v for k, v in out.items() if k != "questions"}
        else:  # pragma: no cover
            parser.error(f"unknown command {args.command}")
            return 2
        print(json.dumps(out, indent=2, default=str))
        return 0
    except MeshError as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        return 1
    finally:
        mesh.close()


if __name__ == "__main__":
    sys.exit(main())
