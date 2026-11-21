"""RDF vocabulary and URI scheme for the knowledge graph.

Instance URIs follow ``urn:mesh:<kind>:<id>`` (e.g. ``urn:mesh:term:customer-identifier``).
Business entities, terms and definitions are keyed by the slug of their label, which is the
cross-source matching rule: "Customer" from a document, a table entity and a definition link
all resolve to ``urn:mesh:entity:customer``.

Layers (named graphs) of a structured source, ``urn:mesh:context:source:<id>:<layer>``:

* ``physical`` tables, columns, primary/unique/foreign keys with ordered column pairs;
* ``semantic`` business entities (``owl:Class``), business terms (``skos:Concept`` + a separate
  ``owl:DatatypeProperty``), named relationships (reified record + ``owl:ObjectProperty``) and
  term relations;
* ``mapping`` table -> entity and column -> term links, relationship -> FK join mappings and the
  schema-level navigational edges derived from FKs.

Documents (``urn:mesh:context:document:<id>``) and captured definitions
(``urn:mesh:context:definition:<id>``) keep their own contexts. Class and property declarations
live in ``config/knowledge_graph_ontology.ttl`` (context ``urn:mesh:context:ontology:core``).
"""

from __future__ import annotations

from rdflib import Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, PROV, RDF, RDFS, SKOS, XSD

MESH = Namespace("urn:mesh:ontology:")

__all__ = ["MESH", "RDF", "RDFS", "SKOS", "XSD", "PROV", "DCTERMS", "OWL", "uri", "context_uri",
           "layer_context_uri", "LAYERS", "STATUSES", "TERM_PREDICATES", "GLOSSARY_SCHEME", "ONTOLOGY_CONTEXT"]

LAYERS = ("physical", "semantic", "mapping")
STATUSES = ("verified", "inferred", "rejected")
# Term-to-term relations the relationship stage may propose (descriptive, never record-level facts).
TERM_PREDICATES = ("expectedAfter", "derivedFrom", "equivalentTo", "dependsOn", "isPartOf")
GLOSSARY_SCHEME = URIRef("urn:mesh:scheme:glossary")
ONTOLOGY_CONTEXT = URIRef("urn:mesh:context:ontology:core")


def uri(kind: str, ident: str) -> URIRef:
    return URIRef(f"urn:mesh:{kind}:{ident}")


def context_uri(kind: str, ident: str) -> URIRef:
    """Named graph holding every assertion derived from one source (document, structured source, definition)."""
    return URIRef(f"urn:mesh:context:{kind}:{ident}")


def layer_context_uri(source_id: str, layer: str) -> URIRef:
    """Named graph of one layer (physical / semantic / mapping) of a structured source."""
    if layer not in LAYERS:
        raise ValueError(f"unknown graph layer {layer}")
    return URIRef(f"urn:mesh:context:source:{source_id}:{layer}")


def entity_uri(slug: str) -> URIRef:
    return uri("entity", slug)


def term_uri(slug: str) -> URIRef:
    return uri("term", slug)


def definition_uri(slug: str) -> URIRef:
    return uri("definition", slug)


def property_uri(term_slug: str) -> URIRef:
    """The OWL datatype property of a business term (kept separate from its SKOS concept)."""
    return uri("property", term_slug)


def predicate_uri(relationship_id: str) -> URIRef:
    """The OWL object property of one named relationship (domain = source entity, range = target)."""
    return uri("predicate", relationship_id)


def term_predicate(name: str) -> URIRef:
    if name not in TERM_PREDICATES:
        raise ValueError(f"unknown term predicate {name}")
    return MESH[name]
